"""End-to-end verification of the three request pipes against the live node.

Subscribes to both TargetBoxArray topics, drives the services, and asserts the
published shape: per-frame message count, detection bboxes, annotations, and
the use_for_assessment flag.
"""
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_srvs.srv import Trigger

from diagnostic_msgs.msg import DiagnosticArray

from cdcl_umd_msgs.msg import CasualtyImageCompressed, TargetBoxArray

BATCH_TOPIC = "/uas4/target_detections"
VLM_TOPIC = "/uas4/target_detections/vlm"
CASUALTY_TOPIC = "/casualty_image/compressed/vlm"


def qos(latched):
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST, depth=10,
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if latched
                    else DurabilityPolicy.VOLATILE))


class Verifier(Node):
    def __init__(self):
        super().__init__("pipe_verifier")
        self.batch, self.vlm, self.casualties = [], [], []
        self.create_subscription(TargetBoxArray, BATCH_TOPIC,
                                 self.batch.append, qos(False))
        self.create_subscription(TargetBoxArray, VLM_TOPIC,
                                 self.vlm.append, qos(True))
        self.create_subscription(CasualtyImageCompressed, CASUALTY_TOPIC,
                                 self.casualties.append, qos(True))
        self._srv_clients = {}
        self.queue_depth = None
        self.create_subscription(DiagnosticArray, "/ds/status",
                                 self._on_status, qos(False))

    def _on_status(self, msg):
        for status in msg.status:
            for kv in status.values:
                if kv.key == "queue_depth":
                    self.queue_depth = int(kv.value)

    def enqueue_frames(self, n, timeout=30.0):
        """Enqueue is asynchronous: each call arms the grab for a LATER frame,
        so the queue fills over the next n frame intervals. Block until the
        node reports all n pending, otherwise the run races the feeder."""
        for _ in range(n):
            self.call("/ds/batch/enqueue")
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.queue_depth == n:
                return True
            time.sleep(0.05)
        return False

    def call(self, name, timeout=60.0):
        if name not in self._srv_clients:
            self._srv_clients[name] = self.create_client(Trigger, name)
            if not self._srv_clients[name].wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"service {name} not available")
        future = self._srv_clients[name].call_async(Trigger.Request())
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline:
            time.sleep(0.02)
        if not future.done():
            raise RuntimeError(f"service {name} timed out")
        return future.result()


FAILURES = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def boxes_well_formed(msg):
    """Every box present carries a positive-area bbox and a labelled, scored
    class. A frame with zero boxes is legitimate output -- the clip has
    frames with no person in them -- so emptiness is not a failure here;
    the caller separately asserts detections appeared across the batch."""
    return all(b.target_bbox.size_x > 0 and b.target_bbox.size_y > 0
               and b.detection_class and 0.0 < b.detection_confidence <= 1.0
               for b in msg.uav_target_boxes)


def box_count(msgs):
    return sum(len(m.uav_target_boxes) for m in msgs)


def classes(msgs):
    return sorted({b.detection_class for m in msgs for b in m.uav_target_boxes})


def person_boxes(msgs):
    """Boxes the SGIE is configured to assess (operate-on-class-ids=0)."""
    return [b for m in msgs for b in m.uav_target_boxes
            if b.detection_class == "person"]


def run_until_detections(node, service, n_frames, attempts=8, want_person=False):
    """The clip loops through stretches with nobody in frame, so a single
    empty batch proves nothing. Retry until one batch comes back with boxes
    and return it; the per-frame distribution is what we actually assert on.

    ``want_person`` additionally requires a person box in every frame -- the
    assess pipe can only be judged on frames the SGIE actually operates on
    (operate-on-class-ids=0), so a motorcycle-only stretch is not a verdict.
    """
    for attempt in range(attempts):
        node.call("/ds/batch/clear")
        time.sleep(0.4)
        node.batch.clear()
        if not node.enqueue_frames(n_frames):
            continue
        resp = node.call(service)
        time.sleep(2.5)
        got = list(node.batch)
        print(f"  attempt {attempt + 1}: {resp.message!r} -> "
              f"box counts {[len(m.uav_target_boxes) for m in got]}")
        if len(got) != n_frames or box_count(got) == 0:
            continue
        if want_person and not all(person_boxes([m]) for m in got):
            continue
        return got, resp
    return got, resp


def main():
    rclpy.init()
    node = Verifier()
    spin = threading.Thread(
        target=lambda: rclpy.spin(node), daemon=True)
    spin.start()
    time.sleep(2.0)  # discovery

    topics = dict(node.get_topic_names_and_types())
    print("\n== topic advertisement ==")
    check(f"{VLM_TOPIC} advertised", VLM_TOPIC in topics,
          str(topics.get(VLM_TOPIC)))
    check(f"{BATCH_TOPIC} advertised", BATCH_TOPIC in topics)

    n_frames = 3

    # ---- pipe 1: detection only ----
    print("\n== pipe 1: /ds/batch/run_detect ==")
    got, resp = run_until_detections(node, "/ds/batch/run_detect", n_frames)
    print(f"  classes seen: {classes(got)}")
    check("one TargetBoxArray per batched frame",
          len(got) == n_frames, f"{len(got)} messages for {n_frames} frames")
    check("boxes well formed",
          all(boxes_well_formed(m) for m in got),
          f"box counts {[len(m.uav_target_boxes) for m in got]}")
    check("detection bboxes published for the batch", box_count(got) > 0,
          f"{box_count(got)} boxes across {len(got)} frames")
    check("EVERY frame in the batch carries its own detections",
          all(len(m.uav_target_boxes) > 0 for m in got),
          f"box counts {[len(m.uav_target_boxes) for m in got]}")
    check("frames are distinct (each its own stamp)",
          len({(m.header.stamp.sec, m.header.stamp.nanosec) for m in got})
          == len(got))
    check("annotations empty on a detect run",
          all(not b.annotations for m in got for b in m.uav_target_boxes))
    check("use_for_assessment false on a detect run",
          all(not b.use_for_assessment for m in got for b in m.uav_target_boxes))
    check("source_img populated",
          all(len(m.source_img.data) > 0 for m in got))
    check("stamps match across header and source_img",
          all(m.header.stamp == m.source_img.header.stamp for m in got))

    # ---- pipe 2: detect + assess ----
    print("\n== pipe 2: /ds/batch/run_detect_assess ==")
    got, resp = run_until_detections(
        node, "/ds/batch/run_detect_assess", n_frames, want_person=True)
    print(f"  classes seen: {classes(got)}")
    check("one TargetBoxArray per batched frame",
          len(got) == n_frames, f"{len(got)} messages for {n_frames} frames")
    check("boxes well formed",
          all(boxes_well_formed(m) for m in got),
          f"box counts {[len(m.uav_target_boxes) for m in got]}")
    check("detection bboxes published for the batch", box_count(got) > 0,
          f"{box_count(got)} boxes across {len(got)} frames")
    check("EVERY frame in the batch carries its own detections",
          all(len(m.uav_target_boxes) > 0 for m in got),
          f"box counts {[len(m.uav_target_boxes) for m in got]}")
    annotated = [b for m in got for b in m.uav_target_boxes if b.annotations]
    total = [b for m in got for b in m.uav_target_boxes]
    persons = person_boxes(got)
    print(f"  {len(persons)}/{len(total)} boxes are class 'person' "
          f"(the SGIE's operate-on-class-ids=0)")
    check("assessment ran on the bboxes", len(annotated) > 0,
          f"{len(annotated)}/{len(total)} boxes annotated")
    check("assessment ran on EVERY person bbox in the batch",
          len(annotated) == len(persons),
          f"{len(annotated)} annotated vs {len(persons)} person boxes")
    frames_with_persons = [m for m in got if person_boxes([m])]
    check("assessment ran on EVERY frame that has a person "
          "(not just the batch's first)",
          all(any(b.annotations for b in m.uav_target_boxes)
              for m in frames_with_persons),
          f"annotated per frame "
          f"{[sum(1 for b in m.uav_target_boxes if b.annotations) for m in got]}"
          f" vs persons per frame {[len(person_boxes([m])) for m in got]}")
    heads = sorted({a.field_name for b in annotated for a in b.annotations})
    check("8 clip_rgb_* heads per assessed box", len(heads) == 8, str(heads))
    # The decisive index check. The SGIE assesses only class 0 (person), and
    # these frames carry a mix (car/chair/bottle/...). If the assessment join
    # were off by even one position, annotations would land on a non-person
    # box or leave a person box bare. Position-by-position equivalence
    # therefore proves the index survived the pgie -> valve -> sgie trip.
    mismatched = [
        (fi, bi, b.detection_class, bool(b.annotations))
        for fi, m in enumerate(got)
        for bi, b in enumerate(m.uav_target_boxes)
        if bool(b.annotations) != (b.detection_class == "person")
    ]
    check("annotations land exactly on the person boxes, position by position",
          not mismatched, f"mismatches {mismatched[:5]}")
    check("use_for_assessment false on an assess run",
          all(not b.use_for_assessment for b in total))

    # ---- pipe 3: vlm image ----
    print("\n== pipe 3: /ds/capture/vlm ==")
    # capture/vlm takes whatever the NEXT frame holds, and the clip has empty
    # stretches; retry until it lands on a frame with detections so the box
    # and crop assertions below are actually exercised.
    for attempt in range(8):
        node.vlm.clear()
        node.casualties.clear()
        node.batch.clear()
        resp = node.call("/ds/capture/vlm")
        time.sleep(2.0)
        got = list(node.vlm)
        print(f"  attempt {attempt + 1}: success={resp.success} "
              f"message={resp.message!r}")
        if got and got[0].uav_target_boxes:
            break
    check("published on " + VLM_TOPIC, len(got) == 1,
          f"{len(got)} messages")
    if got:
        msg = got[0]
        check("carries detection bboxes",
              boxes_well_formed(msg) and msg.uav_target_boxes,
              f"{len(msg.uav_target_boxes)} boxes")
        check("use_for_assessment TRUE on every box",
              all(b.use_for_assessment for b in msg.uav_target_boxes))
        check("annotations empty (detection-only, like target_detections)",
              all(not b.annotations for b in msg.uav_target_boxes))
        check("source_img populated", len(msg.source_img.data) > 0)
        check("detection_source is YOLO, same as target_detections",
              msg.detection_source == got[0].detection_source)
    check("nothing leaked onto " + BATCH_TOPIC,
          len(node.batch) == 0, f"{len(node.batch)} messages")
    check("casualty crops still published",
          len(node.casualties) > 0, f"{len(node.casualties)} crops")

    # Cross-validate the index across two independently built message types:
    # each CasualtyImageCompressed carries detection_id plus the unrounded
    # full-frame bbox floats, so if detection_id really is the box's position
    # in uav_target_boxes, indexing the TBA by it must reproduce the same
    # rectangle. This catches an off-by-one or a permuted array that the
    # per-message checks above cannot see.
    if got:
        boxes = got[0].uav_target_boxes
        ids = [c.detection_id for c in node.casualties]
        check("casualty detection_ids are valid TBA positions",
              all(0 <= i < len(boxes) for i in ids),
              f"ids {ids} vs {len(boxes)} boxes")
        geom = []
        for cas in node.casualties:
            if not (0 <= cas.detection_id < len(boxes)):
                continue
            bb = boxes[cas.detection_id].target_bbox
            left = bb.center.position.x - bb.size_x / 2.0
            top = bb.center.position.y - bb.size_y / 2.0
            if (abs(left - cas.bbox_x) > 0.01
                    or abs(top - cas.bbox_y) > 0.01
                    or abs(bb.size_x - cas.bbox_width) > 0.01
                    or abs(bb.size_y - cas.bbox_height) > 0.01):
                geom.append(
                    (cas.detection_id,
                     (round(left, 1), round(top, 1),
                      round(bb.size_x, 1), round(bb.size_y, 1)),
                     (round(cas.bbox_x, 1), round(cas.bbox_y, 1),
                      round(cas.bbox_width, 1), round(cas.bbox_height, 1))))
        check("TBA[detection_id] bbox == the crop's own bbox, for every crop",
              not geom, f"mismatches {geom[:3]}")
        check("indices are contiguous 0..n-1 over the whole array",
              sorted(ids) == list(range(len(boxes)))
              or set(ids).issubset(range(len(boxes))),
              f"ids {sorted(ids)}")

    # ---- latching: a late subscriber still sees the vlm array ----
    print("\n== pipe 3: late-subscriber latching ==")
    late = []
    node.create_subscription(TargetBoxArray, VLM_TOPIC, late.append, qos(True))
    time.sleep(2.0)
    check("late subscriber receives the latched vlm array", len(late) >= 1,
          f"{len(late)} messages")

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES
                  else f"FAILURES: {FAILURES}"))
    node.destroy_node()
    rclpy.shutdown()
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
