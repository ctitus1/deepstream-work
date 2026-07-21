"""The rclpy node: services, publishers, QoS, callback groups, /ds/status.

DESIGN.md Sec 4 in full. Node name ``ds_pipeline``. Thirteen services in four
callback groups (grp_fast MutuallyExclusive; grp_capture REENTRANT;
grp_record MutuallyExclusive; grp_batch MutuallyExclusive — Sec 4 table), six
publishers with the exact QoS of the topics table, a 1 Hz /ds/status timer
(grp_fast), and the ended-state fail-fast behavior of Sec 5 via
frames.Lifecycle.guard. Spun by a MultiThreadedExecutor(num_threads=8) on the
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
from sensor_msgs.msg import CompressedImage, Image

from cdcl_umd_msgs.msg import (
    AerialDetectionSource,
    Annotation,
    CasualtyImageCompressed,
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


def one_shot_qos(depth: int) -> QoSProfile:
    """RELIABLE, KEEP_LAST depth, TRANSIENT_LOCAL — the latched one-shot
    profile for /mosaic_compressed (depth 5) and /vlm_raw (depth 1), Sec 4."""
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def reliable_qos(depth: int) -> QoSProfile:
    """RELIABLE, KEEP_LAST depth, VOLATILE — /ds/detections and
    /ds/assessments (depth 10), /ds/status (depth 1), Sec 4."""
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
        super().__init__("ds_pipeline")
        self._config = config
        self._lifecycle = lifecycle
        self._grab = grab
        self._recorder = recorder
        self._worker = worker
        self._registry = registry
        self._disk_worker = disk_worker
        self._loop_count_fn = loop_count_fn

        self._seq_lock = threading.Lock()
        self._seq = 0
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
        self._pub_vlm = self.create_publisher(Image, "/vlm_raw", one_shot_qos(1))
        self._pub_preview = self.create_publisher(
            CompressedImage, "/ds/preview/compressed", qos_profile_sensor_data)
        self._pub_detections = self.create_publisher(
            TargetBoxArray, "/ds/detections", reliable_qos(10))
        self._pub_assessments = self.create_publisher(
            CasualtyImageCompressed, "/ds/assessments", reliable_qos(10))
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
        sec, nanosec = timestamps.split_stamp(ntp_ns)
        header.stamp.sec = int(sec)
        header.stamp.nanosec = int(nanosec)
        header.frame_id = self._config.frame_id

    def _set_stamp(self, stamp, ntp_ns: int) -> None:
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

    def _target_box(self, det: batch_pipeline.Detection) -> TargetBox:
        """One TargetBox, field mapping per ros_bridge.py:target_box."""
        bbox = BoundingBox2D()
        bbox.size_x = float(det.width)
        bbox.size_y = float(det.height)
        bbox.center.position.x = float(det.left) + float(det.width) / 2.0
        bbox.center.position.y = float(det.top) + float(det.height) / 2.0

        box = TargetBox()
        box.data_source_id = DATA_SOURCE_ID
        box.target_bbox = bbox
        box.use_for_assessment = True
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

    def publish_detections(self, result: batch_pipeline.FrameResult) -> None:
        """/ds/detections: one TargetBoxArray per batched frame; source_img =
        detections.image_width x image_height JPEG of the frame; all stamps =
        result.item.ntp_ns. Called from the batch worker thread."""
        msg = TargetBoxArray()
        with self._seq_lock:
            msg.seq = self._seq
            self._seq += 1
        self._fill_header(msg.header, result.item.ntp_ns)
        msg.system_id = SYSTEM_ID
        msg.source_img = self._source_image(result)
        msg.gimbal_attitude_quaternion.w = 1.0
        msg.uav_target_boxes = [
            self._target_box(det) for det in result.detections
        ]
        msg.use_for_mosaic = False
        msg.detection_source = AerialDetectionSource.DETECTION_YOLO
        self._pub_detections.publish(msg)

    def publish_assessment(self, result: batch_pipeline.FrameResult,
                           object_id: int) -> None:
        """/ds/assessments: one CasualtyImageCompressed per detected person;
        annotations = the 8 clip_rgb_* heads (ros_bridge.py shape); stamp
        fields from result.item.ntp_ns. Batch worker thread."""
        det = next(
            (d for d in result.detections if d.object_id == object_id), None)
        predictions = result.assessments.get(object_id, {})
        msg = CasualtyImageCompressed()
        msg.data_source_id = DATA_SOURCE_ID
        self._set_stamp(msg.stamp, result.item.ntp_ns)
        msg.image = self._source_image(result)
        self._set_stamp(msg.position.header.stamp, result.item.ntp_ns)
        msg.position.header.frame_id = self._config.frame_id
        msg.annotations = self._annotations(predictions)
        if det is not None:
            msg.bbox_x = float(det.left)
            msg.bbox_y = float(det.top)
            msg.bbox_width = float(det.width)
            msg.bbox_height = float(det.height)
        msg.sensor_frame_id = self._config.frame_id
        msg.platform_name = PLATFORM_NAME
        msg.is_sensor_frame_moving = False
        self._pub_assessments.publish(msg)

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
        """/ds/capture/vlm (Trigger, grp_capture): as mosaic but raw rgb8 on
        /vlm_raw (~11 MB)."""
        def produce(captured: frames.CapturedFrame) -> tuple[bool, str]:
            import numpy as np

            rgb = np.ascontiguousarray(captured.frame_rgba[..., :3])
            height, width = rgb.shape[:2]
            msg = Image()
            self._fill_header(msg.header, captured.ntp_ns)
            msg.height = int(height)
            msg.width = int(width)
            msg.encoding = "rgb8"
            msg.is_bigendian = 0
            msg.step = 3 * int(width)
            msg.data = rgb.tobytes()
            self._pub_vlm.publish(msg)
            return True, _stamp_text(captured.ntp_ns)

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
        assess=False); blocks <= 30 s."""
        return self._answer(response, lambda: self._worker.run_once(assess=False))

    def on_run_detect_assess(self, request, response):
        """/ds/batch/run_detect_assess (Trigger, grp_batch): worker.run_once(
        assess=True); additionally yields /ds/assessments."""
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
