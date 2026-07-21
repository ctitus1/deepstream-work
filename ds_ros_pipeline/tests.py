"""GPU/ROS-free unit tests (DESIGN.md Sec 10 "GPU/ROS-free seams").

Run anywhere, either way: ``python3 -m pytest ds_ros_pipeline/tests.py`` or
``python3 ds_ros_pipeline/tests.py`` (unittest main). Imports only the pure
seams — timestamps, frames (logic classes), source (pts schedule),
infer_configs, disk (DetachSequencer/Recorder finalize with injected
callables), batch_pipeline (crop_bounds, and BatchWorker's publish dispatch
over a faked valve/appsrc). Never imports Gst, pyds, rclpy, or
ros_io/ds_node.

The sys.path insertion below implements the package's flat sibling-import
convention (see the comment atop ds_node.py).
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import frames
import infer_configs
import source
import timestamps
from batch_pipeline import (
    BatchParts,
    BatchWorker,
    Detection,
    ResultCollector,
    crop_bounds,
    indexed_detections,
)
import config
from config import PipelineConfig
from disk import DetachSequencer, Recorder, _Branch
from frames import CaptureKind, GrabState, Lifecycle, Mode
from source import AccessUnit, compute_loop_span, schedule_pts, select_variant
from timestamps import REGISTRY_CAPACITY, TimestampRegistry


class _CopySpy:
    """Injected stand-in for the probe's copy_surface closure (Sec 10)."""

    def __init__(self) -> None:
        self.frames: list[object] = []

    def __call__(self) -> object:
        frame = object()
        self.frames.append(frame)
        return frame

    @property
    def count(self) -> int:
        return len(self.frames)


def _grab_state(**config_overrides) -> tuple[GrabState, Lifecycle]:
    lifecycle = Lifecycle()
    return GrabState(PipelineConfig(**config_overrides), lifecycle), lifecycle


class TestTimestamps(unittest.TestCase):

    def setUp(self) -> None:
        self._saved_helpers = timestamps._stream_helpers
        self.addCleanup(self._restore_helpers)

    def _restore_helpers(self) -> None:
        timestamps._stream_helpers = self._saved_helpers

    def test_registry_bounded_and_lookup(self):
        """TimestampRegistry: stamp/get roundtrip; capacity eviction keeps the
        newest 2048; get on a rolled-out pts returns None."""
        registry = TimestampRegistry()
        stamped = registry.stamp(500)
        self.assertEqual(registry.get(500), stamped)
        self.assertIsNone(registry.get(501))

        self.assertEqual(REGISTRY_CAPACITY, 2048)
        registry = TimestampRegistry()
        overflow = 100
        for pts in range(REGISTRY_CAPACITY + overflow):
            registry.stamp(pts)
        self.assertEqual(len(registry), REGISTRY_CAPACITY)
        for pts in range(overflow):
            self.assertIsNone(registry.get(pts))
        for pts in (overflow, REGISTRY_CAPACITY // 2,
                    REGISTRY_CAPACITY + overflow - 1):
            self.assertIsNotNone(registry.get(pts))

        small = TimestampRegistry(capacity=4)
        for pts in (10, 20, 30, 40, 50, 60):
            small.stamp(pts)
        self.assertEqual(len(small), 4)
        self.assertIsNone(small.get(10))
        self.assertIsNone(small.get(20))
        for pts in (30, 40, 50, 60):
            self.assertIsNotNone(small.get(pts))

    def test_resolve_prefers_stream_ntp_then_registry(self):
        """timestamps.resolve: stream NTP meta wins when wall-clock; registry
        fallback; None when both miss (Sec 5 order)."""
        registry = TimestampRegistry()
        ingest_ns = registry.stamp(500)
        meta = object()

        timestamps._stream_helpers = (
            lambda frame_meta, buffer: ("sender-ntp", 777),
            lambda src: src == "sender-ntp",
        )
        self.assertEqual(timestamps.resolve(registry, 500, frame_meta=meta),
                         777)

        timestamps._stream_helpers = (
            lambda frame_meta, buffer: ("pipeline-clock", 777),
            lambda src: False,
        )
        self.assertEqual(timestamps.resolve(registry, 500, frame_meta=meta),
                         ingest_ns)

        timestamps._stream_helpers = (
            lambda frame_meta, buffer: ("sender-ntp", None),
            lambda src: True,
        )
        self.assertEqual(timestamps.resolve(registry, 500, frame_meta=meta),
                         ingest_ns)

        timestamps._stream_helpers = None  # deepstream_yolo tree unavailable
        self.assertEqual(timestamps.resolve(registry, 500, frame_meta=meta),
                         ingest_ns)

        def _never_called(*_args):
            raise AssertionError("stream helpers consulted without meta")

        timestamps._stream_helpers = (_never_called, _never_called)
        self.assertEqual(timestamps.resolve(registry, 500), ingest_ns)
        self.assertIsNone(timestamps.resolve(registry, 999_999))


class TestSourceSeams(unittest.TestCase):

    @staticmethod
    def _uniform_aus(first_pts: int, count: int, duration: int):
        return [AccessUnit(data=b"x", pts=first_pts + i * duration,
                           duration=duration)
                for i in range(count)]

    def test_pts_schedule_monotonic_unique_across_wraps(self):
        """source.schedule_pts over synthetic AU lists x several n_loops:
        strictly monotonic, globally unique (Sec 5/10)."""
        clip_like = self._uniform_aus(468_700_000, 287, 33_333_333)
        variable = [
            AccessUnit(data=b"x", pts=0, duration=33_000_000),
            AccessUnit(data=b"x", pts=33_000_000, duration=40_000_000),
            AccessUnit(data=b"x", pts=73_000_000, duration=20_000_000),
            AccessUnit(data=b"x", pts=93_000_000, duration=41_000_000),
        ]
        for aus in (clip_like, variable):
            span = compute_loop_span(aus)
            schedule = []
            for n_loops in range(5):
                for au in aus:
                    pts = schedule_pts(au.pts, n_loops, span)
                    self.assertEqual(
                        pts, au.pts + n_loops * span,
                        "pts schedule deviates from the Sec 5 formula "
                        "au.pts + n_loops * loop_span")
                    schedule.append(pts)
            self.assertTrue(
                all(later > earlier
                    for earlier, later in zip(schedule, schedule[1:])),
                "pts schedule regressed across a wrap")
            self.assertEqual(len(set(schedule)), len(schedule),
                             "pts schedule repeated a value across wraps")

        # Uniform list: every consecutive delta -- including the one across
        # each wrap boundary -- equals the frame duration. This is the gapless
        # 30 fps wrap cadence Sec 5 measured; a drifting span (off by even
        # 1 ns/loop) breaks it while staying monotonic and unique.
        span = compute_loop_span(clip_like)
        uniform = [schedule_pts(au.pts, n_loops, span)
                   for n_loops in range(5) for au in clip_like]
        self.assertEqual(
            {later - earlier for earlier, later in zip(uniform, uniform[1:])},
            {33_333_333})

    def test_loop_span_derivation(self):
        """source.compute_loop_span == last_pts - first_pts + duration against
        synthetic AU lists (Sec 3.1)."""
        duration = 33_333_333
        aus = self._uniform_aus(468_700_000, 287, duration)
        self.assertEqual(compute_loop_span(aus),
                         aus[-1].pts - aus[0].pts + duration)

        single = self._uniform_aus(1_000, 1, duration)
        self.assertEqual(compute_loop_span(single), duration)

        # Decode order != presentation order (B-frames): span still derives
        # from the max-pts AU, not the last-pushed one.
        reordered = [
            AccessUnit(data=b"x", pts=0, duration=33),
            AccessUnit(data=b"x", pts=100, duration=33),
            AccessUnit(data=b"x", pts=66, duration=33),
            AccessUnit(data=b"x", pts=33, duration=33),
        ]
        self.assertEqual(compute_loop_span(reordered), 100 - 0 + 33)

        with self.assertRaises(ValueError):
            compute_loop_span([])

    def test_variant_selection(self):
        """source.select_variant: file:// -> file, rtsp(s):// -> rtsp, unknown
        scheme raises (bin construction never touched)."""
        self.assertEqual(select_variant("file:///abs/clip.mp4"), "file")
        self.assertEqual(select_variant("file://streams/clip.mp4"), "file")
        self.assertEqual(select_variant("FILE:///abs/clip.mp4"), "file")
        self.assertEqual(select_variant("rtsp://cam.local/stream"), "rtsp")
        self.assertEqual(select_variant("rtsps://cam.local/stream"), "rtsp")
        for uri in ("http://host/clip.mp4", "shm://socket", "streams/clip.mp4"):
            with self.assertRaises(ValueError):
                select_variant(uri)

        # Registered extension slot, factory dispatch with bins mocked.
        built = object()
        calls: list[tuple] = []

        def factory(uri, config, on_fatal):
            calls.append((uri, config, on_fatal))
            return built

        source.register_variant("Mock", factory)
        self.addCleanup(source._registered_variants.pop, "mock", None)
        self.assertEqual(select_variant("mock://cam0"), "mock")
        config = PipelineConfig(source_uri="mock://cam0")
        self.assertIs(source.create_source(config), built)
        self.assertEqual(calls, [("mock://cam0", config, None)])


class TestGrabState(unittest.TestCase):

    def test_capture_arm_coalesce_and_rearm(self):
        """frames.GrabState: two arms before a frame coalesce to one waiter/one
        consumption; arm after consumption re-arms for the next frame; copy()
        called exactly once per consuming frame (Sec 4 race handling)."""
        state, _ = _grab_state()
        copy = _CopySpy()

        first = state.arm(CaptureKind.MOSAIC)
        second = state.arm(CaptureKind.MOSAIC)
        self.assertIs(first, second)

        state.on_frame(100, 111, copy)
        self.assertEqual(copy.count, 1)
        captured = first.wait(0)
        self.assertIsNotNone(captured)
        self.assertEqual((captured.pts, captured.ntp_ns), (100, 111))
        self.assertIs(captured.frame_rgba, copy.frames[0])
        self.assertIs(second.wait(0), captured)

        # Re-arm after consumption: fresh waiter, next frame serves it.
        rearmed = state.arm(CaptureKind.MOSAIC)
        self.assertIsNot(rearmed, first)
        self.assertFalse(rearmed._event.is_set())
        state.on_frame(133, 222, copy)
        self.assertEqual(copy.count, 2)
        self.assertEqual(rearmed.wait(0).pts, 133)

        # All armed kinds are served from the same frame with ONE copy.
        mosaic = state.arm(CaptureKind.MOSAIC)
        snap_raw = state.arm(CaptureKind.SNAPSHOT_RAW)
        self.assertIsNot(mosaic, snap_raw)
        state.on_frame(166, 333, copy)
        self.assertEqual(copy.count, 3)
        self.assertIs(mosaic.wait(0).frame_rgba, copy.frames[2])
        self.assertIs(snap_raw.wait(0).frame_rgba, copy.frames[2])

        # Idle frame: nothing armed, no copy.
        state.on_frame(200, 444, copy)
        self.assertEqual(copy.count, 3)

        # Unresolvable stamp: frame skipped, arming survives to the next frame.
        pending = state.arm(CaptureKind.SNAPSHOT_PNG)
        with self.assertLogs("ds_ros_pipeline.frames", level="WARNING"):
            state.on_frame(233, None, copy)
        self.assertEqual(copy.count, 3)
        self.assertFalse(pending._event.is_set())
        state.on_frame(266, 555, copy)
        self.assertEqual(copy.count, 4)
        self.assertEqual(pending.wait(0).pts, 266)
        self.assertEqual(state.counters()["resolve_skips"], 1)

    def test_enqueue_counter_does_not_coalesce(self):
        """N request_enqueue between frames consume the next N distinct frames;
        success=False at batch_capacity (Sec 4)."""
        state, _ = _grab_state(batch_capacity=4)
        copy = _CopySpy()

        self.assertEqual(state.request_enqueue(), (True, 1))
        self.assertEqual(state.request_enqueue(), (True, 2))
        self.assertEqual(state.request_enqueue(), (True, 3))

        for index in range(3):
            state.on_frame(1000 + index * 10, 42 + index, copy)
        self.assertEqual(copy.count, 3)
        self.assertEqual(state.pending_depth(), 3)
        items = state.swap_pending()
        self.assertEqual([item.pts for item in items], [1000, 1010, 1020])
        self.assertEqual([item.ntp_ns for item in items], [42, 43, 44])
        self.assertEqual([item.frame_rgba for item in items], copy.frames)

        # Counter drained: further frames are not consumed.
        state.on_frame(1030, 45, copy)
        self.assertEqual(copy.count, 3)
        self.assertEqual(state.pending_depth(), 0)

        # Capacity: depth counts queued + not-yet-consumed pending requests.
        for expected_depth in (1, 2, 3, 4):
            self.assertEqual(state.request_enqueue(), (True, expected_depth))
        self.assertEqual(state.request_enqueue(), (False, 4))
        for index in range(4):
            state.on_frame(2000 + index, 50 + index, copy)
        self.assertEqual(state.pending_depth(), 4)
        self.assertEqual(state.request_enqueue(), (False, 4))

        self.assertEqual(state.clear(), 4)
        self.assertEqual(state.pending_depth(), 0)
        self.assertEqual(state.request_enqueue(), (True, 1))

    def test_enqueue_depth_counts_in_flight_copy(self):
        """A request_enqueue racing the copy window (frame consumed but not yet
        appended) sees the in-flight frame in the depth: no over-admission at
        the capacity boundary (frames.request_enqueue contract, Sec 4)."""
        state, _ = _grab_state(batch_capacity=2)
        copy = _CopySpy()

        self.assertEqual(state.request_enqueue(), (True, 1))
        state.on_frame(100, 1, copy)
        self.assertEqual(state.pending_depth(), 1)

        mid_copy: list[tuple[bool, int]] = []

        def reentrant_copy() -> object:
            # Runs outside the mutex by design (Sec 4), so this models a
            # service-thread request landing mid-copy: one frame queued plus
            # this in-flight one already fill capacity 2.
            mid_copy.append(state.request_enqueue())
            return copy()

        self.assertEqual(state.request_enqueue(), (True, 2))
        state.on_frame(133, 2, reentrant_copy)
        self.assertEqual(mid_copy, [(False, 2)])
        self.assertEqual(state.pending_depth(), 2)
        self.assertEqual(state.counters()["enqueue_drops"], 0)
        self.assertEqual([item.pts for item in state.swap_pending()],
                         [100, 133])

    def test_snapshot_and_swap(self):
        """swap_pending returns the snapshot; enqueues during a 'run' land in the
        fresh deque and survive; clear empties only the pending deque (Sec 4/6)."""
        state, _ = _grab_state()
        copy = _CopySpy()

        for pts in (100, 133):
            self.assertTrue(state.request_enqueue()[0])
            state.on_frame(pts, pts * 2, copy)
        snapshot = state.swap_pending()
        self.assertEqual([item.pts for item in snapshot], [100, 133])
        self.assertEqual(state.pending_depth(), 0)

        # Enqueue "during the run": lands in the fresh deque, survives.
        self.assertTrue(state.request_enqueue()[0])
        state.on_frame(166, 332, copy)
        self.assertEqual(state.pending_depth(), 1)
        self.assertEqual([item.pts for item in snapshot], [100, 133])

        # clear touches only the pending deque, never the snapshot.
        self.assertEqual(state.clear(), 1)
        self.assertEqual(state.pending_depth(), 0)
        self.assertEqual(len(snapshot), 2)
        self.assertEqual(state.swap_pending(), [])

    def test_continuous_stride_auto_enqueue(self):
        """Mode.DETECT auto-enqueues every stride-th frame; switching modes is
        mutually exclusive (Sec 4)."""
        state, _ = _grab_state(continuous_stride=3)
        copy = _CopySpy()

        self.assertIs(state.set_mode(Mode.DETECT), Mode.OFF)
        self.assertIs(state.mode, Mode.DETECT)
        # 8 frames, deliberately NOT a multiple of the stride: the counter is
        # left mid-cycle (8 % 3 == 2), so the post-switch immediate fire below
        # can only pass if set_mode really resets the stride clock.
        for index in range(8):
            state.on_frame(1000 + index, 1, copy)
        self.assertEqual([item.pts for item in state.swap_pending()],
                         [1000, 1003, 1006])

        # Turning the other mode on turns this one off (single mode slot)
        # and resets the stride clock: the next frame fires immediately.
        self.assertIs(state.set_mode(Mode.DETECT_ASSESS), Mode.DETECT)
        self.assertIs(state.mode, Mode.DETECT_ASSESS)
        state.on_frame(2000, 1, copy)
        self.assertEqual([item.pts for item in state.swap_pending()], [2000])

        # Same-mode set is a no-op: the stride clock keeps running.
        self.assertIs(state.set_mode(Mode.DETECT_ASSESS), Mode.DETECT_ASSESS)
        for pts in (2001, 2002):
            state.on_frame(pts, 1, copy)
        self.assertEqual(state.pending_depth(), 0)
        state.on_frame(2003, 1, copy)
        self.assertEqual([item.pts for item in state.swap_pending()], [2003])

        # Manual enqueue on a stride-fire frame: at most one item per frame.
        self.assertIs(state.set_mode(Mode.DETECT), Mode.DETECT_ASSESS)
        before = copy.count
        self.assertTrue(state.request_enqueue()[0])
        state.on_frame(3000, 1, copy)
        self.assertEqual(copy.count, before + 1)
        self.assertEqual([item.pts for item in state.swap_pending()], [3000])

        # OFF stops auto-enqueue.
        self.assertIs(state.set_mode(Mode.OFF), Mode.DETECT)
        before = copy.count
        for pts in (4000, 4001, 4002):
            state.on_frame(pts, 1, copy)
        self.assertEqual(copy.count, before)
        self.assertEqual(state.pending_depth(), 0)

        # Deque at capacity: continuous mode skips and counts, never blocks.
        tight, _ = _grab_state(continuous_stride=1, batch_capacity=1)
        tight.set_mode(Mode.DETECT)
        tight.on_frame(1, 1, copy)
        tight.on_frame(2, 1, copy)
        self.assertEqual(tight.pending_depth(), 1)
        self.assertEqual(tight.counters()["continuous_skips"], 1)

    def test_ended_state_machine(self):
        """frames.Lifecycle: guard fail-fast set (capture/snapshot/enqueue/
        record_start) vs always-allowed (clear/mode/run/record_stop); armed
        waiters wake with None on notify_ended (Sec 5)."""
        fail_fast = ("capture", "snapshot", "enqueue", "record_start")
        always_allowed = ("clear", "mode", "run", "record_stop")

        lifecycle = Lifecycle()
        self.assertEqual(lifecycle.state, Lifecycle.RUNNING)
        for service in fail_fast + always_allowed:
            self.assertIsNone(lifecycle.guard(service))

        lifecycle.mark_ended()
        self.assertEqual(lifecycle.state, Lifecycle.ENDED)
        lifecycle.mark_ended()  # idempotent
        self.assertEqual(lifecycle.state, Lifecycle.ENDED)
        for service in fail_fast:
            self.assertEqual(lifecycle.guard(service), frames.ENDED_MESSAGE)
        for service in always_allowed:
            self.assertIsNone(lifecycle.guard(service))

        # Armed waiters wake with None on notify_ended.
        state, live_lifecycle = _grab_state()
        waiter = state.arm(CaptureKind.MOSAIC)
        self.assertFalse(waiter._event.is_set())
        live_lifecycle.mark_ended()
        state.notify_ended()
        self.assertTrue(waiter._event.is_set())
        self.assertIsNone(waiter.wait(0))

        # Arming after ended returns an already-resolved None waiter.
        late = state.arm(CaptureKind.SNAPSHOT_RAW)
        self.assertTrue(late._event.is_set())
        self.assertIsNone(late.wait(0))


class TestCropBounds(unittest.TestCase):

    @staticmethod
    def _det(left, top, width, height) -> Detection:
        return Detection(left=left, top=top, width=width, height=height,
                         confidence=0.9, class_id=0, label="person",
                         object_id=0)

    def test_integer_box_passes_through(self):
        """batch_pipeline.crop_bounds: an in-frame integer bbox maps to its
        own pixel bounds; a frame-covering box maps to the whole frame."""
        self.assertEqual(crop_bounds(self._det(10, 20, 30, 40), 2560, 1440),
                         (10, 20, 40, 60))
        self.assertEqual(crop_bounds(self._det(0, 0, 2560, 1440), 2560, 1440),
                         (0, 0, 2560, 1440))

    def test_fractional_box_covers_partial_pixels(self):
        """Fractional edges floor left/top and ceil right/bottom, so every
        partially covered pixel lands in the crop."""
        self.assertEqual(
            crop_bounds(self._det(10.4, 20.6, 30.2, 40.2), 2560, 1440),
            (10, 20, 41, 61))

    def test_clamped_to_frame(self):
        """Boxes overhanging any edge clamp to the frame bounds."""
        self.assertEqual(crop_bounds(self._det(-5.0, -8.0, 20.0, 30.0), 640, 480),
                         (0, 0, 15, 22))
        self.assertEqual(crop_bounds(self._det(630.0, 470.0, 20.0, 30.0), 640, 480),
                         (630, 470, 640, 480))

    def test_degenerate_and_outside_boxes_return_none(self):
        """Non-positive size or a box entirely outside the frame yields None
        (publisher falls back to the full-frame image)."""
        self.assertIsNone(crop_bounds(self._det(10, 10, 0, 30), 640, 480))
        self.assertIsNone(crop_bounds(self._det(10, 10, 30, -1), 640, 480))
        self.assertIsNone(crop_bounds(self._det(640, 100, 20, 20), 640, 480))
        self.assertIsNone(crop_bounds(self._det(100, 480, 20, 20), 640, 480))
        self.assertIsNone(crop_bounds(self._det(-50, 100, 20, 20), 640, 480))


class TestIndexedDetections(unittest.TestCase):
    """batch_pipeline.indexed_detections: the array-position contract.

    Position i of uav_target_boxes must be the detection DeepStream indexed
    i, because assessments and casualty detection_id both key on that index.
    """

    @staticmethod
    def _det(object_id, label="person") -> Detection:
        return Detection(left=object_id, top=0, width=10, height=10,
                         confidence=0.9, class_id=0, label=label,
                         object_id=object_id)

    def test_already_ordered_passes_through_without_complaint(self):
        dets = [self._det(i) for i in range(4)]
        ordered, complaint = indexed_detections(tuple(dets))
        self.assertEqual([d.object_id for d in ordered], [0, 1, 2, 3])
        self.assertIsNone(complaint)

    def test_permuted_input_is_restored_to_index_order(self):
        """A permuted probe emission must not shift the array: box i stays
        detection i, otherwise assessments[i] would annotate the wrong box."""
        dets = [self._det(2), self._det(0), self._det(3), self._det(1)]
        ordered, complaint = indexed_detections(tuple(dets))
        self.assertEqual([d.object_id for d in ordered], [0, 1, 2, 3])
        self.assertIsNone(complaint)

    def test_empty_frame_is_not_a_complaint(self):
        ordered, complaint = indexed_detections(())
        self.assertEqual(ordered, [])
        self.assertIsNone(complaint)

    def test_non_contiguous_indices_complain_but_stay_ordered(self):
        """A gap means position != detection_id; the join by object_id is
        still right, so the caller warns rather than dropping data."""
        dets = [self._det(0), self._det(2)]
        ordered, complaint = indexed_detections(tuple(dets))
        self.assertEqual([d.object_id for d in ordered], [0, 2])
        self.assertIsNotNone(complaint)
        self.assertIn("not contiguous", complaint)


class TestBatchPublishDispatch(unittest.TestCase):
    """Which publish callbacks each of the three pipes fires (Sec 4).

    BatchWorker._execute_run is the single dispatch point; it only touches
    Gst through _push_items and parts.v_assess, both faked here, so the
    pipe->callback mapping is testable without a GPU.
    """

    class _FakeValve:
        def __init__(self) -> None:
            self.drop_history: list[bool] = []

        def set_property(self, name, value) -> None:
            assert name == "drop"
            self.drop_history.append(value)

    def setUp(self) -> None:
        self.valve = self._FakeValve()
        self.calls: list[str] = []
        self.item = frames.BatchItem(frame_rgba=object(), ntp_ns=1_000, pts=7)
        collector = ResultCollector()
        parts = BatchParts(pipeline=None, src_batch=None, v_assess=self.valve,
                           pgie=None, sgie=None)
        grab, _ = _grab_state()
        self.worker = BatchWorker(
            PipelineConfig(), grab, parts, collector,
            publish_detections=lambda r: self.calls.append("detections"),
            publish_assessments=lambda r: self.calls.append("assessments"),
            publish_casualties=lambda r: self.calls.append("casualties"),
            publish_vlm_detections=lambda r: self.calls.append("vlm"))
        # Stand in for the appsrc push + probe round-trip: feed the collector
        # the detections the pgie probe would have produced for this pts.
        detections = (Detection(left=1, top=2, width=3, height=4,
                                confidence=0.9, class_id=0, label="person",
                                object_id=0),)

        def fake_push(items) -> bool:
            for item in items:
                collector.add_detections(item.pts, detections)
                collector.add_assessments(item.pts, {0: {"probabilities": [1.0]}})
            return True

        self.worker._push_items = fake_push

    def test_detect_only_publishes_detections(self):
        """assess=False -> publish_detections alone; the valve stays dropping
        so the sgie never sees the frame."""
        success, _ = self.worker._execute_run([self.item], assess=False,
                                              wait_timeout=1.0)
        self.assertTrue(success)
        self.assertEqual(self.calls, ["detections"])
        self.assertEqual(self.valve.drop_history, [True, True])

    def test_detect_assess_publishes_assessments_only(self):
        """assess=True -> publish_assessments alone (the plain detection
        array is NOT additionally published), valve opened for the run and
        restored after."""
        success, message = self.worker._execute_run([self.item], assess=True,
                                                    wait_timeout=1.0)
        self.assertTrue(success)
        self.assertEqual(self.calls, ["assessments"])
        self.assertEqual(self.valve.drop_history, [False, True])
        self.assertIn("assessed", message)

    def test_capture_vlm_publishes_both_vlm_boxes_and_casualties(self):
        """run_capture -> the /uas4/target_detections/vlm TargetBoxArray AND
        the casualty crops, detection-only (valve dropping throughout)."""
        success, _ = self.worker.run_capture(self.item, timeout=1.0)
        self.assertTrue(success)
        self.assertEqual(self.calls, ["vlm", "casualties"])
        self.assertEqual(self.valve.drop_history, [True, True])

    def test_capture_vlm_rejected_in_continuous_mode(self):
        """Continuous mode owns the batch pipeline; the vlm pipe refuses
        rather than interleaving a run."""
        self.worker.set_continuous(True, assess=False)
        success, message = self.worker.run_capture(self.item, timeout=1.0)
        self.assertFalse(success)
        self.assertEqual(message, "continuous mode active")
        self.assertEqual(self.calls, [])

    def test_every_batched_frame_gets_its_own_message(self):
        """Per frame, not per batch: k queued frames -> k publish calls."""
        items = [frames.BatchItem(frame_rgba=object(), ntp_ns=1_000 + i,
                                  pts=100 + i)
                 for i in range(6)]
        success, message = self.worker._execute_run(items, assess=False,
                                                    wait_timeout=1.0)
        self.assertTrue(success)
        self.assertEqual(self.calls, ["detections"] * 6)
        self.assertIn("6/6 frames", message)


class TestDetachSequencer(unittest.TestCase):

    @staticmethod
    def _sequencer(log: list, wait_drain):
        def install_idle_probe(callback):
            log.append("install_idle_probe")
            callback()  # tests call the IDLE probe inline (Sec 10)

        return DetachSequencer(
            install_idle_probe=install_idle_probe,
            unlink=lambda: log.append("unlink"),
            send_eos=lambda: log.append("send_eos"),
            release_pad=lambda: log.append("release_pad"),
            wait_drain=wait_drain,
            finalize=lambda: log.append("finalize"),
        )

    def test_detach_sequencer_ordering(self):
        """disk.DetachSequencer: exact order unlink -> send_eos -> release_pad
        inside the probe callback; wait_drain strictly after; finalize runs in
        both outcomes; no drain wait before unlink+release (Sec 7/10)."""
        log: list = []

        def wait_drain(timeout):
            log.append(("wait_drain", timeout))
            return True

        drained = self._sequencer(log, wait_drain).stop(2.5)
        self.assertTrue(drained)
        self.assertEqual(log, [
            "install_idle_probe",
            "unlink",
            "send_eos",
            "release_pad",
            ("wait_drain", 2.5),
            "finalize",
        ])

        # Same exact ordering when the drain fails; finalize still runs.
        log_failed: list = []

        def wait_drain_failed(timeout):
            log_failed.append(("wait_drain", timeout))
            return False

        drained = self._sequencer(log_failed, wait_drain_failed).stop(2.5)
        self.assertFalse(drained)
        self.assertEqual(log_failed, [
            "install_idle_probe",
            "unlink",
            "send_eos",
            "release_pad",
            ("wait_drain", 2.5),
            "finalize",
        ])

    def test_detach_sequencer_timeout_force_finalize(self):
        """A drain_done that never fires: stop returns drained=False within the
        timeout and finalize still ran (Sec 7 step 4)."""
        log: list = []
        never_fires = threading.Event()
        sequencer = self._sequencer(log, never_fires.wait)
        started = time.monotonic()
        drained = sequencer.stop(0.05)
        elapsed = time.monotonic() - started
        self.assertFalse(drained)
        self.assertLess(elapsed, 2.0)
        self.assertEqual(log[:4], [
            "install_idle_probe", "unlink", "send_eos", "release_pad"])
        self.assertEqual(log[-1], "finalize")

    def test_force_finalize_disposable_thread(self):
        """Recorder._finalize_branch: drained => NULL+remove inline on the
        calling thread; not drained => on the disposable 'rec-force-finalize'
        thread, finalize_pending cleared once it completes (Sec 7 step 4)."""
        recorder = Recorder(PipelineConfig(record_stop_timeout=2.0),
                            live=object(), registry=TimestampRegistry())
        finalized: list[tuple[str, object]] = []
        recorder._null_and_remove = (
            lambda branch: finalized.append(
                (threading.current_thread().name, branch)))

        def make_branch() -> _Branch:
            return _Branch(elements=[], tee_pad=None, q_rec=None,
                           sink_rec=None, path=Path("/tmp/rec_test.ts"),
                           sidecar_path=Path("/tmp/rec_test.jsonl"))

        drained_branch = make_branch()
        drained_branch.drain_done.set()
        recorder._finalize_branch(drained_branch)
        self.assertEqual(finalized,
                         [(threading.current_thread().name, drained_branch)])
        self.assertFalse(recorder._finalize_pending)

        wedged_branch = make_branch()  # drain_done never fired
        recorder._finalize_branch(wedged_branch)
        self.assertEqual(len(finalized), 2)
        self.assertEqual(finalized[1], ("rec-force-finalize", wedged_branch))
        self.assertFalse(recorder._finalize_pending)


class TestInferConfigs(unittest.TestCase):

    _B1_TEMPLATE = """\
[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=/workspace/deepstream-work/models/yolo12x_640_640x384.onnx
model-engine-file=/workspace/deepstream-work/models/yolo12x_640_640x384.onnx_b1_gpu0_fp16.engine
labelfile-path=/workspace/deepstream-work/models/yolo12x_640_640x384.labels.txt
batch-size=1
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=/workspace/deepstream-work/lib/libnvdsinfer_custom_impl_Yolo.so
output-blob-names=output
maintain-aspect-ratio=1
symmetric-padding=1
cluster-mode=2

[class-attrs-all]
pre-cluster-threshold=0.2
nms-iou-threshold=0.45
topk=300
"""

    _GOLDEN_B8 = """\
[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=/workspace/deepstream-work/models/yolo12x_640_640x384.onnx
model-engine-file=/workspace/deepstream-work/models/yolo12x_640_640x384.onnx_b8_gpu0_fp16.engine
labelfile-path=/workspace/deepstream-work/models/yolo12x_640_640x384.labels.txt
batch-size=8
network-mode=2
num-detected-classes=80
interval=0
gie-unique-id=1
process-mode=1
network-type=0
parse-bbox-func-name=NvDsInferParseYolo
custom-lib-path=/workspace/deepstream-work/lib/libnvdsinfer_custom_impl_Yolo.so
output-blob-names=output
maintain-aspect-ratio=1
symmetric-padding=1
cluster-mode=2

[class-attrs-all]
pre-cluster-threshold=2.0
nms-iou-threshold=0.45
topk=300

[class-attrs-0]
pre-cluster-threshold=0.4
nms-iou-threshold=0.45
topk=300
"""

    def _sections(self, text):
        return {name: dict(entries) for name, entries
                in infer_configs._parse_sections(text, "rendered")}

    def test_person_only_class_thresholds(self):
        """Every class but person gets an unreachable threshold, so nvinfer
        discards those detections during parsing and no object meta is ever
        created for them — downstream may assume every Detection is a person.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / infer_configs.TEMPLATE_RELPATH
            template.parent.mkdir(parents=True)
            template.write_text(self._B1_TEMPLATE)

            sections = self._sections(
                infer_configs.render_batch_yolo_config(8, root, 0.4))
            # Confidence is a probability, so >1.0 can never be met.
            self.assertGreater(
                float(sections["class-attrs-all"]["pre-cluster-threshold"]), 1.0)
            self.assertEqual(
                sections["class-attrs-0"]["pre-cluster-threshold"], "0.4")
            # The person section must not silently inherit a different
            # clustering policy than the template's.
            for key in ("nms-iou-threshold", "topk"):
                self.assertEqual(sections["class-attrs-0"][key],
                                 sections["class-attrs-all"][key])

    def test_min_confidence_is_settable(self):
        """detect.min_confidence lands in the person section verbatim."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / infer_configs.TEMPLATE_RELPATH
            template.parent.mkdir(parents=True)
            template.write_text(self._B1_TEMPLATE)

            for value, expected in ((0.4, "0.4"), (0.25, "0.25"),
                                    (0.9, "0.9"), (1.0, "1")):
                sections = self._sections(
                    infer_configs.render_batch_yolo_config(8, root, value))
                self.assertEqual(
                    sections["class-attrs-0"]["pre-cluster-threshold"],
                    expected)

    def test_config_default_min_confidence_is_0_4(self):
        self.assertEqual(PipelineConfig().detect_min_confidence, 0.4)
        self.assertIn("detect.min_confidence", config.PARAMETER_MAP)

    def test_infer_config_golden(self):
        """infer_configs.render_batch_yolo_config(8) matches the golden config
        (b1 settings + batch-size=8 + b8 engine path + the person-only class
        thresholds, Sec 6/10)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / infer_configs.TEMPLATE_RELPATH
            template.parent.mkdir(parents=True)
            template.write_text(self._B1_TEMPLATE)

            rendered = infer_configs.render_batch_yolo_config(8, root)
            self.assertEqual(rendered, self._GOLDEN_B8)

            # b1 no longer round-trips to the template: the class filter is
            # applied at every batch size. [property] is still untouched.
            b1 = infer_configs.render_batch_yolo_config(1, root)
            self.assertEqual(
                dict(infer_configs._property_entries(
                    infer_configs._parse_sections(b1, "b1"), "b1")),
                dict(infer_configs._property_entries(
                    infer_configs._parse_sections(self._B1_TEMPLATE, "t"), "t")))
            self.assertIn("[class-attrs-0]", b1)

            with self.assertRaises(ValueError):
                infer_configs.render_batch_yolo_config(0, root)
            # Confidence must be a probability.
            for bad in (0.0, -0.1, 1.5):
                with self.assertRaises(ValueError):
                    infer_configs.render_batch_yolo_config(
                        8, root, min_confidence=bad)

        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                infer_configs.render_batch_yolo_config(8, Path(empty))

        self.assertEqual(
            infer_configs.batch_yolo_config_path(8).name,
            "ds_ros_infer_batch_yolo12x_640_640x384_b8.txt")

        # The real repo template, when present, renders with only the two
        # overridden keys changed relative to its own parse.
        repo_template = (Path(__file__).resolve().parents[1]
                         / infer_configs.TEMPLATE_RELPATH)
        if repo_template.is_file():
            root = Path(__file__).resolve().parents[1]
            rendered = infer_configs.render_batch_yolo_config(8, root)
            parsed = dict(infer_configs._property_entries(
                infer_configs._parse_sections(rendered, "rendered"),
                "rendered"))
            template_props = dict(infer_configs._property_entries(
                infer_configs._parse_sections(repo_template.read_text(),
                                              repo_template),
                repo_template))
            self.assertEqual(parsed["batch-size"], "8")
            self.assertEqual(
                parsed["model-engine-file"],
                f"{template_props['onnx-file']}_b8_gpu0_fp16.engine")
            for key, value in template_props.items():
                if key not in ("batch-size", "model-engine-file"):
                    self.assertEqual(parsed[key], value)


if __name__ == "__main__":
    unittest.main()
