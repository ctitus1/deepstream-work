"""Dump the ACTUAL published messages for the three pipes, field by field.

Prints real values (topic, seq, class, bbox, use_for_assessment, annotation
heads) rather than assertions, then applies the pass/fail checks at the end
so the verdict and the raw evidence can be compared against each other.
"""
import os
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from std_srvs.srv import Trigger

from diagnostic_msgs.msg import DiagnosticArray

from cdcl_umd_msgs.msg import TargetBoxArray

BATCH = "/uas4/target_detections"
VLM = "/uas4/target_detections/vlm"
# pgie is person-only at or above detect.min_confidence (config default 0.4).
MIN_CONFIDENCE = float(os.environ.get("DS_MIN_CONFIDENCE", "0.4"))
FAIL = []


def qos(latched=False):
    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST, depth=20,
        durability=(DurabilityPolicy.TRANSIENT_LOCAL if latched
                    else DurabilityPolicy.VOLATILE))


def check(label, cond, detail=""):
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    if not cond:
        FAIL.append(label)


class N(Node):
    def __init__(self):
        super().__init__("evidence")
        self.batch, self.vlm = [], []
        self.depth = None
        self.create_subscription(TargetBoxArray, BATCH, self.batch.append, qos())
        self.create_subscription(TargetBoxArray, VLM, self.vlm.append, qos(True))
        self.create_subscription(DiagnosticArray, "/ds/status", self._st, qos())
        self._cl = {}

    def _st(self, m):
        for s in m.status:
            for kv in s.values:
                if kv.key == "queue_depth":
                    self.depth = int(kv.value)

    def call(self, name, timeout=60.0):
        if name not in self._cl:
            self._cl[name] = self.create_client(Trigger, name)
            self._cl[name].wait_for_service(timeout_sec=10.0)
        f = self._cl[name].call_async(Trigger.Request())
        end = time.time() + timeout
        while not f.done() and time.time() < end:
            time.sleep(0.02)
        return f.result()

    def fill(self, n, timeout=30.0):
        self.call("/ds/batch/clear")
        time.sleep(0.4)
        for _ in range(n):
            self.call("/ds/batch/enqueue")
        end = time.time() + timeout
        while time.time() < end:
            if self.depth == n:
                return True
            time.sleep(0.05)
        return False


def dump(msg, topic, limit=6):
    print(f"    topic={topic} seq={msg.seq} "
          f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} "
          f"frame_id={msg.header.frame_id!r}")
    print(f"    source_img: {len(msg.source_img.data)} bytes "
          f"format={msg.source_img.format!r} | "
          f"use_for_mosaic={msg.use_for_mosaic} "
          f"detection_source={msg.detection_source}")
    print(f"    uav_target_boxes: {len(msg.uav_target_boxes)}")
    for i, b in enumerate(msg.uav_target_boxes[:limit]):
        bb = b.target_bbox
        left = bb.center.position.x - bb.size_x / 2.0
        top = bb.center.position.y - bb.size_y / 2.0
        heads = [a.field_name for a in b.annotations]
        print(f"      [{i}] {b.detection_class:<14} conf={b.detection_confidence:.3f}"
              f" bbox=({left:7.1f},{top:7.1f},{bb.size_x:6.1f},{bb.size_y:6.1f})"
              f" use_for_assessment={str(b.use_for_assessment):<5}"
              f" annotations={len(b.annotations)}")
        if heads:
            probs = {a.field_name.replace('clip_rgb_', ''):
                     [round(v, 3) for v in a.observation] for a in b.annotations}
            print(f"           heads={probs}")
    if len(msg.uav_target_boxes) > limit:
        print(f"      ... {len(msg.uav_target_boxes) - limit} more")


def persons(msgs):
    return [b for m in msgs for b in m.uav_target_boxes
            if b.detection_class == "person"]


def gate(label, msgs):
    """The pgie is person-only at or above MIN_CONFIDENCE; anything else in
    a message means the class/confidence gate is not doing its job."""
    boxes = [b for m in msgs for b in m.uav_target_boxes]
    others = sorted({b.detection_class for b in boxes
                     if b.detection_class != "person"})
    check(f"{label}: every detection is class 'person'", not others,
          f"also saw {others}" if others else f"{len(boxes)} boxes, all person")
    weak = [round(b.detection_confidence, 3) for b in boxes
            if b.detection_confidence < MIN_CONFIDENCE]
    check(f"{label}: every detection >= min_confidence ({MIN_CONFIDENCE})",
          not weak, f"below: {weak}" if weak else
          f"lowest {min((b.detection_confidence for b in boxes), default=0):.3f}")


def main():
    rclpy.init()
    node = N()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    time.sleep(2.0)
    n = 3

    # ============ PIPE 1: detect ============
    print("\n" + "=" * 72)
    print("PIPE 1 -- /ds/batch/run_detect")
    print("=" * 72)
    for _ in range(8):
        node.batch.clear()
        node.fill(n)
        r1 = node.call("/ds/batch/run_detect")
        time.sleep(2.5)
        g1 = list(node.batch)
        if len(g1) == n and sum(len(m.uav_target_boxes) for m in g1):
            break
    print(f"  service response: success={r1.success} message={r1.message!r}")
    for i, m in enumerate(g1):
        print(f"\n  --- TargetBoxArray {i + 1} of {len(g1)} ---")
        dump(m, BATCH)
    print("\n  VERDICT:")
    check("published a TargetBoxArray per batched frame", len(g1) == n,
          f"{len(g1)} arrays for {n} frames")
    check("arrays carry uav_target_boxes",
          all(m.uav_target_boxes for m in g1),
          f"counts {[len(m.uav_target_boxes) for m in g1]}")
    gate("detect", g1)
    check("NO annotations on any box (detect only)",
          all(not b.annotations for m in g1 for b in m.uav_target_boxes))
    check("use_for_assessment=false on every box",
          all(not b.use_for_assessment for m in g1 for b in m.uav_target_boxes))

    # ============ PIPE 2: detect + assess ============
    print("\n" + "=" * 72)
    print("PIPE 2 -- /ds/batch/run_detect_assess")
    print("=" * 72)
    for _ in range(8):
        node.batch.clear()
        node.fill(n)
        r2 = node.call("/ds/batch/run_detect_assess")
        time.sleep(2.5)
        g2 = list(node.batch)
        if len(g2) == n and all(persons([m]) for m in g2):
            break
    print(f"  service response: success={r2.success} message={r2.message!r}")
    for i, m in enumerate(g2):
        print(f"\n  --- TargetBoxArray {i + 1} of {len(g2)} ---")
        dump(m, BATCH)
    print("\n  VERDICT:")
    check("published a TargetBoxArray per batched frame", len(g2) == n,
          f"{len(g2)} arrays for {n} frames")
    check("arrays carry uav_target_boxes",
          all(m.uav_target_boxes for m in g2),
          f"counts {[len(m.uav_target_boxes) for m in g2]}")
    gate("detect+assess", g2)
    ann = [b for m in g2 for b in m.uav_target_boxes if b.annotations]
    check("annotations present", len(ann) > 0, f"{len(ann)} annotated boxes")
    check("every person box annotated with all 8 clip_rgb_* heads",
          len(ann) == len(persons(g2))
          and all(len(b.annotations) == 8 for b in ann),
          f"{len(ann)} annotated vs {len(persons(g2))} persons")
    check("annotations land exactly on person boxes, position by position",
          all(bool(b.annotations) == (b.detection_class == "person")
              for m in g2 for b in m.uav_target_boxes))
    check("every frame in the batch has its own annotated box",
          all(any(b.annotations for b in m.uav_target_boxes) for m in g2),
          f"per frame {[sum(1 for b in m.uav_target_boxes if b.annotations) for m in g2]}")

    # ============ PIPE 3: vlm ============
    print("\n" + "=" * 72)
    print("PIPE 3 -- /ds/capture/vlm")
    print("=" * 72)
    for _ in range(8):
        node.vlm.clear()
        node.batch.clear()
        r3 = node.call("/ds/capture/vlm")
        time.sleep(2.0)
        g3 = list(node.vlm)
        if g3 and g3[0].uav_target_boxes:
            break
    print(f"  service response: success={r3.success} message={r3.message!r}")
    for i, m in enumerate(g3):
        print(f"\n  --- TargetBoxArray {i + 1} of {len(g3)} ---")
        dump(m, VLM)
    print("\n  VERDICT:")
    check("published exactly one TargetBoxArray on " + VLM, len(g3) == 1,
          f"{len(g3)} arrays")
    check("array carries uav_target_boxes",
          bool(g3 and g3[0].uav_target_boxes),
          f"{len(g3[0].uav_target_boxes) if g3 else 0} boxes")
    gate("vlm", g3)
    check("NO annotations on any box",
          all(not b.annotations for m in g3 for b in m.uav_target_boxes))
    check("use_for_assessment=TRUE on every box",
          all(b.use_for_assessment for m in g3 for b in m.uav_target_boxes))
    check("nothing published on " + BATCH, len(node.batch) == 0,
          f"{len(node.batch)} arrays")

    print("\n" + "=" * 72)
    print("ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}")
    print("=" * 72)
    node.destroy_node()
    rclpy.shutdown()
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
