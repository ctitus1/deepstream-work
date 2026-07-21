"""The rclpy node: services, publishers, QoS, callback groups, /ds/status.

DESIGN.md Sec 4, amended: the batch outputs live on ONE shared
TargetBoxArray topic, /uas4/target_detections (ros_bridge.py's detect-topic
name) — detect runs publish plain boxes, assess runs the same boxes with
annotations filled on the assessed ones. /ds/capture/vlm runs detection over
the captured frame and publishes the SAME TargetBoxArray a detect run would,
with use_for_assessment=True on every box, on the sibling topic
/uas4/target_detections/vlm. /casualty_image/compressed/vlm and /vlm_raw no
longer exist. Node name ``ds_pipeline``. Thirteen services in
four callback groups (grp_fast MutuallyExclusive; grp_capture REENTRANT;
grp_record MutuallyExclusive; grp_batch MutuallyExclusive — Sec 4 table),
five publishers with the exact QoS of the README topics table, a 1 Hz
/ds/status timer (grp_fast), and the ended-state fail-fast behavior of Sec 5
via frames.Lifecycle.guard. Spun by a MultiThreadedExecutor(num_threads=8) on the
"ros" thread (Sec 2) — the executor is owned by ds_node.

Message builders borrow the field mapping of src/ros_bridge.py
(copy-one-stamp-everywhere: every header.stamp and nested stamp = the same
resolved ntp_ns split). rclpy / message-package imports are module-level here
(this module is only imported in-container); everything logic-bearing lives
in the sibling modules so unit tests never import this file.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_srvs.srv import SetBool, Trigger

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from sensor_msgs.msg import CompressedImage

from cdcl_umd_msgs.msg import (
    AerialDetectionSource,
    Annotation,
    TargetBox,
    TargetBoxArray,
)
from vision_msgs.msg import BoundingBox2D

import batch_pipeline
import disk
import frames
import timestamps
from config import PipelineConfig
from frames import CaptureKind, Mode

CAPTURE_TIMEOUT_S = 2.0  # Sec 4: capture/snapshot block <= ~2 s for a frame
SNAPSHOT_WRITE_TIMEOUT_S = 10.0  # bound the disk-worker wait (PNG ~0.2-0.6 s)
MOSAIC_JPEG_QUALITY = 90     # Sec 4: full-res one-shot, quality 90
DETECTION_JPEG_QUALITY = 85  # matches deepstream_yolo.pipeline default

# ros_bridge.py field-mapping defaults (no metadata source in this process).
SYSTEM_ID = 0
DATA_SOURCE_ID = 0
PLATFORM_NAME = "deepstream"


def no_type_description_service() -> list:
    """Parameter overrides that keep the node from advertising
    ``~/get_type_description`` (Jazzy+; declared read-only at node init, so a
    construction-time override is the only way to turn it off). The service
    is optional introspection nothing here uses, and a humble-era peer — the
    ros-humble foxglove bridge in particular — cannot resolve its type and
    logs a WARN on every graph poll. On distros without the service the
    override is simply never consumed."""
    from rclpy.parameter import Parameter

    return [Parameter("start_type_description_service", value=False)]


def one_shot_qos(depth: int) -> QoSProfile:
    """RELIABLE, KEEP_LAST depth, TRANSIENT_LOCAL — the latched profile for
    /mosaic_compressed (depth 5) and the capture/vlm TargetBoxArray
    (depth 10), so a subscriber attaching after the call still gets it."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def reliable_qos(depth: int) -> QoSProfile:
    """RELIABLE, KEEP_LAST depth, VOLATILE — /uas4/target_detections
    (depth 10) and /ds/status (depth 1)."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.VOLATILE,
    )


def _encode_jpeg(frame_rgba, quality: int,
                 size: tuple[int, int] | None = None) -> bytes:
    """RGBA numpy -> JPEG bytes (optionally resized to (width, height))."""
    import cv2

    bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
    if size is not None:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return encoded.tobytes()


def _stamp_text(ntp_ns: int) -> str:
    """The service-response representation of a stamp (test 2/12: comparable
    to the published header.stamp)."""
    sec, nanosec = timestamps.split_stamp(ntp_ns)
    return f"{sec}.{nanosec:09d}"


class DsRosNode(Node):
    """``ds_pipeline`` node: pure I/O shell over the injected collaborators.

    Collaborators (constructed and wired by ds_node): config, lifecycle,
    grab (frames.GrabState), recorder (disk.Recorder), worker
    (batch_pipeline.BatchWorker), registry, disk_worker, and
    ``loop_count_fn`` (-> the feeder's n_loops, Sec 3.1) for /ds/status.

    Every service callback body: check lifecycle.guard(...) first (Sec 5
    fail-fast — ended captures return success=false "source ended"
    immediately, no 2 s wait), then delegate to the collaborator, then format
    Trigger/SetBool responses per the Sec 4 semantics column. Blocking
    behavior per group: grp_fast callbacks never block; grp_capture blocks up
    to CAPTURE_TIMEOUT_S (+ disk-worker wait for snapshots); grp_record's
    stop blocks up to record.stop_timeout; grp_batch blocks up to 30 s.
    """

    def __init__(self, config: PipelineConfig, lifecycle: frames.Lifecycle,
                 grab: frames.GrabState, recorder: disk.Recorder,
                 worker: batch_pipeline.BatchWorker,
                 registry: timestamps.TimestampRegistry,
                 disk_worker: disk.DiskWorker,
                 loop_count_fn: Callable[[], int]) -> None:
        """Declare parameters were already read (config); create the four
        callback groups, all publishers (exact QoS above +
        qos_profile_sensor_data for /ds/preview/compressed), all thirteen
        services, and the 1 Hz status timer. Main thread, before spin."""
        super().__init__("ds_pipeline",
                         parameter_overrides=no_type_description_service())
        self._config = config
        self._lifecycle = lifecycle
        self._grab = grab
        self._recorder = recorder
        self._worker = worker
        self._registry = registry
        self._disk_worker = disk_worker
        self._loop_count_fn = loop_count_fn

        self._seq_lock = threading.Lock()
        # TargetBoxArray.seq, counted per TBA topic: a consumer watching
        # /uas4/target_detections for gaps must not see phantom ones because
        # a capture/vlm run consumed a number from a shared counter.
        self._seq = {"batch": 0, "vlm": 0}
        # Coalesced-call publish-once bookkeeping (Sec 4 race handling): per
        # kind, one mutex serializing produce/publish and the last waiter
        # already served, so N callers sharing a waiter yield ONE message and
        # identical response messages.
        self._capture_locks = {kind: threading.Lock() for kind in CaptureKind}
        self._capture_results: dict[CaptureKind, tuple[object, str]] = {}

        self._grp_fast = MutuallyExclusiveCallbackGroup()
        self._grp_capture = ReentrantCallbackGroup()
        self._grp_record = MutuallyExclusiveCallbackGroup()
        self._grp_batch = MutuallyExclusiveCallbackGroup()

        self._pub_mosaic = self.create_publisher(
            CompressedImage, "/mosaic_compressed", one_shot_qos(5))
        self._pub_preview = self.create_publisher(
            CompressedImage, "/ds/preview/compressed", qos_profile_sensor_data)
        # One shared TargetBoxArray topic (ros_bridge.py's detect-topic name):
        # detect runs publish plain boxes, assess runs the same boxes with
        # annotations filled on the assessed ones.
        self._pub_tba = self.create_publisher(
            TargetBoxArray, "/uas4/target_detections", reliable_qos(10))
        # capture/vlm TargetBoxArray: message-identical to a detect run's
        # except use_for_assessment=True on every box. Latched (unlike the
        # batch topic) because this one is a one-shot, like every other
        # one-shot output here — a subscriber attaching after the service
        # call still receives it.
        self._pub_tba_vlm = self.create_publisher(
            TargetBoxArray, "/uas4/target_detections/vlm", one_shot_qos(10))
        self._pub_status = self.create_publisher(
            DiagnosticArray, "/ds/status", reliable_qos(1))

        for srv_type, name, callback, group in (
            (Trigger, "/ds/capture/mosaic", self.on_capture_mosaic, self._grp_capture),
            (Trigger, "/ds/capture/vlm", self.on_capture_vlm, self._grp_capture),
            (Trigger, "/ds/snapshot", self.on_snapshot, self._grp_capture),
            (Trigger, "/ds/snapshot_raw", self.on_snapshot_raw, self._grp_capture),
            (Trigger, "/ds/batch/enqueue", self.on_enqueue, self._grp_fast),
            (Trigger, "/ds/batch/clear", self.on_clear, self._grp_fast),
            (SetBool, "/ds/mode/continuous_detect", self.on_continuous_detect, self._grp_fast),
            (SetBool, "/ds/mode/continuous_detect_assess", self.on_continuous_detect_assess, self._grp_fast),
            (Trigger, "/ds/batch/run_detect", self.on_run_detect, self._grp_batch),
            (Trigger, "/ds/batch/run_detect_assess", self.on_run_detect_assess, self._grp_batch),
            (Trigger, "/ds/record/start", self.on_record_start, self._grp_record),
            (Trigger, "/ds/record/start_raw", self.on_record_start_raw, self._grp_record),
            (Trigger, "/ds/record/stop", self.on_record_stop, self._grp_record),
        ):
            self.create_service(srv_type, name, callback, callback_group=group)

        self._status_timer = self.create_timer(
            1.0, self.on_status_timer, callback_group=self._grp_fast)

    # -- shared helpers ------------------------------------------------------

    def _answer(self, response, body: Callable[[], tuple[bool, str]]):
        """Run ``body`` and always fill the response (Sec 4: every service
        answers) — an unexpected exception becomes success=false, never an
        unanswered call."""
        try:
            success, message = body()
        except Exception as exc:  # noqa: BLE001 — must answer regardless
            self.get_logger().error(f"service failed: {exc!r}")
            success, message = False, f"error: {exc}"
        response.success = success
        response.message = message
        return response

    def _fill_header(self, header, ntp_ns: int) -> None:
        self._fill_stamp(header.stamp, ntp_ns)
        header.frame_id = self._config.frame_id

    def _fill_stamp(self, stamp, ntp_ns: int) -> None:
        sec, nanosec = timestamps.split_stamp(ntp_ns)
        stamp.sec = int(sec)
        stamp.nanosec = int(nanosec)

    def _capture(self, kind: CaptureKind, guard_name: str,
                 produce: Callable[[frames.CapturedFrame], tuple[bool, str]],
                 ) -> tuple[bool, str]:
        """Common one-shot flow (Sec 4): guard -> arm -> wait <= 2 s ->
        produce exactly once per waiter (coalesced callers reuse the first
        caller's result: one message, same stamp/path for everyone)."""
        reason = self._lifecycle.guard(guard_name)
        if reason is not None:
            return False, reason
        waiter = self._grab.arm(kind)
        captured = waiter.wait(CAPTURE_TIMEOUT_S)
        if captured is None:
            if self._lifecycle.state == frames.Lifecycle.ENDED:
                return False, frames.ENDED_MESSAGE
            return False, "timed out waiting for a frame"
        with self._capture_locks[kind]:
            cached = self._capture_results.get(kind)
            if cached is not None and cached[0] is waiter:
                return True, cached[1]
            success, message = produce(captured)
            if success:
                self._capture_results[kind] = (waiter, message)
            return success, message

    def _source_image(self, result: batch_pipeline.FrameResult) -> CompressedImage:
        """detections.image_width x image_height JPEG of the batched frame
        (message-size hygiene, Sec 4), stamped with the item's ntp_ns."""
        config = self._config
        jpeg = _encode_jpeg(
            result.item.frame_rgba, DETECTION_JPEG_QUALITY,
            size=(config.detections_image_width, config.detections_image_height))
        msg = CompressedImage()
        self._fill_header(msg.header, result.item.ntp_ns)
        msg.format = "jpeg"
        msg.data = jpeg
        return msg

    def _bbox_scale(self, result: batch_pipeline.FrameResult,
                    ) -> tuple[float, float]:
        """(x, y) factors mapping source-pixel detections onto source_img.

        nvinfer reports boxes in FULL source-frame pixels (2560x1440 for the
        default clip), but source_img is a detections.image_width x
        image_height JPEG (640x368). Publishing the two together unscaled
        means any consumer that overlays the boxes on the image it was handed
        — Foxglove's Image panel included — draws them roughly 4x too large
        and off-frame. ros_bridge.py has no such gap: its bboxes and its
        source_img are already the same space.

        x and y are scaled independently because _source_image resizes to
        exactly (width, height) without preserving aspect: 2560x1440 (1.78)
        into 640x368 (1.74) is a slight vertical squash, and a single uniform
        factor would leave boxes progressively misplaced down the frame.
        """
        height, width = result.item.frame_rgba.shape[:2]
        if not width or not height:
            return 1.0, 1.0
        return (self._config.detections_image_width / float(width),
                self._config.detections_image_height / float(height))

    def _target_box(self, det: batch_pipeline.Detection,
                    scale: tuple[float, float],
                    use_for_assessment: bool = False) -> TargetBox:
        """One TargetBox, field mapping per ros_bridge.py:target_box.

        ``scale`` is _bbox_scale's (x, y): the box is emitted in source_img
        pixel coordinates, not source-frame ones, so it lines up with the
        image published alongside it.

        ``use_for_assessment`` ("flag if a target should be considered for
        assessment", TargetBox.msg) marks the box for a downstream assessor.
        Only the capture/vlm path sets it — the batch detect/assess paths
        publish boxes already carrying whatever assessment they ran, so
        flagging them for more would be a request nobody meant to make.
        """
        scale_x, scale_y = scale
        left = float(det.left) * scale_x
        top = float(det.top) * scale_y
        width = float(det.width) * scale_x
        height = float(det.height) * scale_y

        bbox = BoundingBox2D()
        bbox.size_x = width
        bbox.size_y = height
        bbox.center.position.x = left + width / 2.0
        bbox.center.position.y = top + height / 2.0

        box = TargetBox()
        box.data_source_id = DATA_SOURCE_ID
        box.target_bbox = bbox
        box.use_for_assessment = bool(use_for_assessment)
        box.detection_source.detection_source = AerialDetectionSource.DETECTION_YOLO
        box.detection_class = str(det.label)
        box.detection_confidence = float(det.confidence)
        return box

    def _annotations(self, predictions: dict[str, dict]) -> list[Annotation]:
        """The 8 clip_rgb_* heads, per ros_bridge.py:annotations."""
        annotations = []
        for name, prediction in sorted(predictions.items()):
            annotation = Annotation()
            annotation.field_name = f"clip_rgb_{name}"
            annotation.observation = [
                float(value) for value in prediction["probabilities"]
            ]
            annotations.append(annotation)
        return annotations

    # -- publishers used from outside the node (thread-safe, Sec 2) ---------

    def publish_preview(self, jpeg: bytes, ntp_ns: int) -> None:
        """/ds/preview/compressed (sensor-data QoS): CompressedImage
        format='jpeg', header from ntp_ns + config.frame_id. Called from the
        preview branch's Gst streaming thread (live_pipeline.connect_preview
        binds this). Non-blocking."""
        msg = CompressedImage()
        self._fill_header(msg.header, ntp_ns)
        msg.format = "jpeg"
        msg.data = jpeg
        self._pub_preview.publish(msg)

    def _target_box_array(self, result: batch_pipeline.FrameResult,
                          boxes: list[TargetBox],
                          seq_key: str = "batch",
                          do_assessment: bool = False) -> TargetBoxArray:
        """TargetBoxArray shell for the TBA topics: header.stamp and
        source_img (the detections.image_width x image_height JPEG of the
        frame, its own header included) all carry the frame's resolved
        ingest time. ``seq_key`` selects the per-topic seq counter.

        ``boxes`` are expected in source_img pixel coordinates (_bbox_scale),
        so the array and the image it carries share one space.

        ``do_assessment`` is the array-level request flag, set only by the
        capture/vlm path — the array-wide counterpart of the per-box
        use_for_assessment those same boxes carry."""
        msg = TargetBoxArray()
        with self._seq_lock:
            msg.seq = self._seq[seq_key]
            self._seq[seq_key] += 1
        self._fill_header(msg.header, result.item.ntp_ns)
        msg.system_id = SYSTEM_ID
        msg.source_img = self._source_image(result)
        msg.gimbal_attitude_quaternion.w = 1.0
        msg.uav_target_boxes = boxes
        msg.use_for_mosaic = False
        msg.do_assessment = bool(do_assessment)
        msg.detection_source = AerialDetectionSource.DETECTION_YOLO
        return msg

    def _indexed_detections(self, result: batch_pipeline.FrameResult,
                            ) -> list[batch_pipeline.Detection]:
        """``result.detections`` in DeepStream index order, so the box at
        position i of uav_target_boxes is the detection DeepStream indexed i.

        Thin logging wrapper over batch_pipeline.indexed_detections, which
        holds the ordering rule itself (and its rationale) in a module unit
        tests can import.
        """
        detections, complaint = batch_pipeline.indexed_detections(
            result.detections)
        if complaint is not None:
            self.get_logger().warning(complaint)
        return detections

    def publish_detections(self, result: batch_pipeline.FrameResult) -> None:
        """run_detect / continuous_detect output: one TargetBoxArray per
        batched frame on /uas4/target_detections — detections only (every
        box's annotations empty) plus the compressed frame image; all stamps
        = result.item.ntp_ns. Called from the batch worker thread."""
        scale = self._bbox_scale(result)
        boxes = [self._target_box(det, scale)
                 for det in self._indexed_detections(result)]
        self._pub_tba.publish(self._target_box_array(result, boxes))

    def publish_assessments(self, result: batch_pipeline.FrameResult) -> None:
        """run_detect_assess / continuous output: one TargetBoxArray per
        frame on /uas4/target_detections — the SAME boxes a detect run would
        publish, with the 8 clip_rgb_* heads (ros_bridge.py shape) filled
        into the annotations of every box the injury SGIE assessed; all
        stamps = result.item.ntp_ns. Batch worker thread.

        The annotations land on the right box because both sides of the join
        use the index det_collect stamped into the object meta: the box at
        position i IS detection i, and assessments[i] is the tensor output
        the SGIE produced for that same object."""
        scale = self._bbox_scale(result)
        boxes = []
        for det in self._indexed_detections(result):
            box = self._target_box(det, scale)
            predictions = result.assessments.get(det.object_id)
            if predictions:
                box.annotations = self._annotations(predictions)
            boxes.append(box)
        self._pub_tba.publish(self._target_box_array(result, boxes))

    def publish_vlm_detections(self, result: batch_pipeline.FrameResult) -> None:
        """capture/vlm output: one TargetBoxArray on
        /uas4/target_detections/vlm — field-for-field what publish_detections
        would put on /uas4/target_detections for the same frame (same boxes,
        empty annotations, same source_img, all stamps = result.item.ntp_ns),
        except every box carries use_for_assessment=True and the array itself
        carries do_assessment=True. Batch worker thread."""
        scale = self._bbox_scale(result)
        boxes = [self._target_box(det, scale, use_for_assessment=True)
                 for det in self._indexed_detections(result)]
        self._pub_tba_vlm.publish(
            self._target_box_array(result, boxes, seq_key="vlm",
                                   do_assessment=True))

    # -- service callbacks (executor threads, groups as annotated) ----------

    def on_capture_mosaic(self, request, response):
        """/ds/capture/mosaic (Trigger, grp_capture): arm MOSAIC, wait <= 2 s,
        JPEG-encode full-res q90 on THIS thread, publish once latched on
        /mosaic_compressed; message = the stamp used. Coalesced calls share
        one message/stamp (Sec 4)."""
        def produce(captured: frames.CapturedFrame) -> tuple[bool, str]:
            jpeg = _encode_jpeg(captured.frame_rgba, MOSAIC_JPEG_QUALITY)
            msg = CompressedImage()
            self._fill_header(msg.header, captured.ntp_ns)
            msg.format = "jpeg"
            msg.data = jpeg
            self._pub_mosaic.publish(msg)
            return True, _stamp_text(captured.ntp_ns)

        return self._answer(
            response, lambda: self._capture(CaptureKind.MOSAIC, "capture", produce))

    def on_capture_vlm(self, request, response):
        """/ds/capture/vlm (Trigger, grp_capture): arm VLM, wait <= 2 s for
        the next frame, run DETECTION over just that frame via
        worker.run_capture (bypasses the pending queue; serialized with
        other runs; rejected while a continuous mode is on), and publish the
        detect-run TargetBoxArray with use_for_assessment=True on
        /uas4/target_detections/vlm. Blocks until published
        (capture wait + <= 10 s run). Coalesced calls share one frame and
        one run (Sec 4); message = the stamp used + run counts."""
        def produce(captured: frames.CapturedFrame) -> tuple[bool, str]:
            item = frames.BatchItem(frame_rgba=captured.frame_rgba,
                                    ntp_ns=captured.ntp_ns,
                                    pts=captured.pts)
            success, message = self._worker.run_capture(item)
            return success, f"{_stamp_text(captured.ntp_ns)} ({message})"

        return self._answer(
            response, lambda: self._capture(CaptureKind.VLM, "capture", produce))

    def on_enqueue(self, request, response):
        """/ds/batch/enqueue (Trigger, grp_fast): grab.request_enqueue();
        message = resulting depth; success=false if queue full. Sub-ms."""
        def body() -> tuple[bool, str]:
            reason = self._lifecycle.guard("enqueue")
            if reason is not None:
                return False, reason
            ok, depth = self._grab.request_enqueue()
            return ok, str(depth)

        return self._answer(response, body)

    def on_clear(self, request, response):
        """/ds/batch/clear (Trigger, grp_fast): grab.clear() — pending deque
        only (Sec 4)."""
        return self._answer(
            response, lambda: (True, f"cleared {self._grab.clear()} frames"))

    def on_run_detect(self, request, response):
        """/ds/batch/run_detect (Trigger, grp_batch): worker.run_once(
        assess=False) — detection TargetBoxArrays on /uas4/target_detections;
        blocks <= 30 s."""
        return self._answer(response, lambda: self._worker.run_once(assess=False))

    def on_run_detect_assess(self, request, response):
        """/ds/batch/run_detect_assess (Trigger, grp_batch): worker.run_once(
        assess=True) — assessment TargetBoxArrays (annotations filled) on
        /uas4/target_detections; the plain detection arrays are not
        additionally published."""
        return self._answer(response, lambda: self._worker.run_once(assess=True))

    def _set_continuous(self, active: bool, assess: bool) -> tuple[bool, str]:
        """Shared toggle body (Sec 4): turning either mode on turns the other
        off (grab.set_mode does that atomically); off restores Mode.OFF and
        discards pending auto-enqueued frames so a later manual run starts
        from a clean queue (a run already in flight still completes)."""
        mode = Mode.DETECT_ASSESS if assess else Mode.DETECT
        if active:
            self._grab.set_mode(mode)
            self._worker.set_continuous(True, assess)
            return True, f"continuous {mode.value} on"
        self._grab.set_mode(Mode.OFF)
        self._worker.set_continuous(False, assess)
        discarded = self._grab.clear()
        return True, f"continuous {mode.value} off ({discarded} pending frames discarded)"

    def on_continuous_detect(self, request, response):
        """/ds/mode/continuous_detect (SetBool, grp_fast): grab.set_mode +
        worker.set_continuous; turning either mode on turns the other off."""
        return self._answer(
            response, lambda: self._set_continuous(request.data, assess=False))

    def on_continuous_detect_assess(self, request, response):
        """/ds/mode/continuous_detect_assess (SetBool, grp_fast): same with
        the valve open."""
        return self._answer(
            response, lambda: self._set_continuous(request.data, assess=True))

    def _record_start(self, raw: bool) -> tuple[bool, str]:
        reason = self._lifecycle.guard("record_start")
        if reason is not None:
            return False, reason
        return self._recorder.start(raw=raw)

    def on_record_start(self, request, response):
        """/ds/record/start (Trigger, grp_record): recorder.start(raw=False);
        message = file path; success=false if recording or ended."""
        return self._answer(response, lambda: self._record_start(raw=False))

    def on_record_start_raw(self, request, response):
        """/ds/record/start_raw (Trigger, grp_record): raw I420 variant."""
        return self._answer(response, lambda: self._record_start(raw=True))

    def on_record_stop(self, request, response):
        """/ds/record/stop (Trigger, grp_record): recorder.stop(); message =
        path, frames written/dropped, drained=true|false (Sec 4); idempotent
        after source-EOS finalization."""
        def body() -> tuple[bool, str]:
            success, message, stats = self._recorder.stop()
            if stats is not None:
                message = (
                    f"path={stats.path}"
                    f" frames_written={stats.frames_written}"
                    f" frames_dropped={stats.frames_dropped}"
                    f" drained={'true' if stats.drained else 'false'}")
            return success, message

        return self._answer(response, body)

    def _snapshot(self, raw: bool) -> tuple[bool, str]:
        kind = CaptureKind.SNAPSHOT_RAW if raw else CaptureKind.SNAPSHOT_PNG

        def produce(captured: frames.CapturedFrame) -> tuple[bool, str]:
            output_dir = Path(self._config.snapshot_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = disk.snapshot_path(output_dir, captured.ntp_ns, raw=raw)
            writer = disk.write_ppm_with_sidecar if raw else disk.write_png
            done = self._disk_worker.submit(partial(
                writer, captured.frame_rgba, path, captured.pts,
                captured.ntp_ns))
            if not done.wait(SNAPSHOT_WRITE_TIMEOUT_S):
                return False, f"snapshot write timed out ({path})"
            if not path.exists():
                return False, f"snapshot write failed ({path})"
            return True, str(path)

        return self._capture(kind, "snapshot", produce)

    def on_snapshot(self, request, response):
        """/ds/snapshot (Trigger, grp_capture): arm SNAPSHOT_PNG, wait for
        the frame, submit write_png to the disk worker, block until the file
        is closed; message = path."""
        return self._answer(response, lambda: self._snapshot(raw=False))

    def on_snapshot_raw(self, request, response):
        """/ds/snapshot_raw (Trigger, grp_capture): .ppm + .json sidecar."""
        return self._answer(response, lambda: self._snapshot(raw=True))

    def on_status_timer(self) -> None:
        """1 Hz /ds/status (DiagnosticArray, Sec 4): state running|ended,
        mode, queue depth, recording state, drop counters, loop count
        (loop_count_fn). grp_fast; non-blocking."""
        state = self._lifecycle.state
        counters = self._grab.counters()
        values = {
            "state": state,
            "mode": self._grab.mode.value,
            "queue_depth": str(self._grab.pending_depth()),
            "recording": self._recorder.state,
            "loop_count": str(self._loop_count_fn()),
        }
        values.update({key: str(count) for key, count in counters.items()})

        status = DiagnosticStatus()
        status.level = (DiagnosticStatus.OK
                        if state == frames.Lifecycle.RUNNING
                        else DiagnosticStatus.WARN)
        status.name = "ds_pipeline"
        status.message = state
        status.hardware_id = self._config.frame_id
        status.values = [
            KeyValue(key=key, value=value) for key, value in values.items()
        ]

        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._config.frame_id
        msg.status = [status]
        self._pub_status.publish(msg)
