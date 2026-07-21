# ros_io.py review notes

## Coder round 1

Implemented `ros_io.py` completely per DESIGN.md Sec 4/5/8 within the pinned
skeleton interfaces (no signature changes):

- **QoS**: `one_shot_qos` (RELIABLE / KEEP_LAST depth / TRANSIENT_LOCAL) for
  `/mosaic_compressed` (5) and `/vlm_raw` (1); `reliable_qos` (VOLATILE) for
  `/ds/detections`, `/ds/assessments` (10) and `/ds/status` (1);
  `qos_profile_sensor_data` for `/ds/preview/compressed`.
- **Groups**: exactly the Sec 4 table — `grp_fast`/`grp_record`/`grp_batch`
  MutuallyExclusive, `grp_capture` Reentrant; all 13 services created with the
  right group; 1 Hz status timer in `grp_fast`. Executor (`MultiThreadedExecutor
  num_threads=8`) is owned by ds_node per the skeleton docstring — ros_io only
  defines the groups/callbacks (nothing here blocks the constructor).
- **Capture flow** (`_capture` helper): `lifecycle.guard` first (ended =>
  immediate `success=false "source ended"`, no 2 s wait — verified <0.1 s), then
  `grab.arm` + `waiter.wait(2.0)`. Coalesced-caller publish-once: a per-kind
  lock + "last waiter served" cache — N callers sharing one waiter produce ONE
  topic message and identical response messages (Sec 4 race semantics; verified
  with two racing threads). Timeout answers `success=false` within ~2 s.
- **Mosaic**: full-res JPEG q90 encoded on the service thread (cv2, lazy
  import), published latched; response message = `sec.nanosec` of the stamp
  used (== header.stamp, test 2 comparable). **VLM**: raw `Image` rgb8 built
  from numpy without cv_bridge — `step=3*W`, `data=rgb.tobytes()` (verified
  step 192 / len 192*H on a synthetic frame).
- **Snapshots**: arm SNAPSHOT_PNG/RAW, submit `disk.write_png` /
  `write_ppm_with_sidecar` to the DiskWorker, block on the done event (bounded
  10 s) and verify the file exists before answering; message = path. mkdir of
  `snapshot.output_dir` happens before submit.
- **Record**: start/start_raw guard + delegate to `recorder.start`; stop
  formats `path=… frames_written=… frames_dropped=… drained=true|false` from
  `RecordStats` (idempotent-after-finalize handled inside Recorder). Blocking
  bound (stop_timeout + force-finalize margin) lives in Recorder; the callback
  only ever waits on it.
- **Batch/mode/fast**: enqueue message = resulting depth (1..N back-to-back,
  verified), ended fail-fast on enqueue; clear reports removed count;
  run_detect/_assess delegate to `worker.run_once(assess=…)`; continuous
  toggles call `grab.set_mode` (either-on-turns-other-off is in GrabState) +
  `worker.set_continuous(active, assess)`.
- **Message builders** borrow `src/ros_bridge.py` field mapping verbatim:
  TargetBoxArray (seq counter under lock, system_id, gimbal quaternion w=1,
  source_img = 640x368 JPEG q85 — the deepstream_yolo default quality —
  use_for_mosaic=False, DETECTION_YOLO, TargetBox center-from-corner bbox
  math, use_for_assessment=True); CasualtyImageCompressed (data_source_id,
  stamp + position.header.stamp + image header all = the same split ntp_ns,
  8 sorted `clip_rgb_*` Annotations with `probabilities` as observation,
  bbox_x/y/width/height from the matching Detection, platform_name
  "deepstream", is_sensor_frame_moving=False). Copy-one-stamp-everywhere
  discipline throughout (`_fill_header`/`_set_stamp` from
  `timestamps.split_stamp`).
- **/ds/status**: DiagnosticArray, one DiagnosticStatus (level OK/WARN by
  state, message = state) with KeyValues: state, mode, queue_depth, recording
  (Recorder.state), loop_count (loop_count_fn), and the four GrabState drop
  counters.
- **Always answers**: every callback runs through `_answer`, which converts an
  unexpected exception into `success=false, message="error: …"` instead of an
  unanswered call (verified).

Validation:
- `python3 -m py_compile ds_ros_pipeline/ros_io.py` clean.
- In-container (`deepstream-work:ds-ros`, ros2_ws mounted, no GPU needed):
  module import + QoS profile assertions + JPEG/stamp helper checks pass.
- In-container node-level smoke test (fake Recorder/BatchWorker/DiskWorker +
  real GrabState/Lifecycle/TimestampRegistry, publishers swapped for
  recorders): asserted the 13-service inventory and group assignment
  (grp_capture Reentrant), mosaic coalescing (2 racing callers -> 1 message,
  identical stamps), vlm rgb8 shape/step, snapshot PNG/PPM+json files written
  under /tmp, capture timeout <3 s, enqueue depth 1..3, clear, run_detect
  delegation, continuous toggle transitions, record stop message format,
  TargetBoxArray/CasualtyImageCompressed field mapping incl. annotation
  ordering, preview publish, status KeyValues, ended-state fail-fast timing,
  and `_answer` exception containment. Output: `NODE_SMOKE_OK`.

Notes for testers:
- Constants borrowed from ros_bridge defaults where this process has no
  metadata source: SYSTEM_ID=0, DATA_SOURCE_ID=0, PLATFORM_NAME="deepstream";
  sensor_frame_id reuses config.frame_id. Trivial to lift into config.py later
  if the real values differ.
- `continuous_*` with `data=false` unconditionally sets Mode.OFF (Sec 4 says
  "data=false: stop" without a which-mode qualifier); turning off a mode that
  was not the active one therefore also stops the other. Flagging in case the
  master tester prefers a matching-mode no-op.

## Interface notes

None — the skeleton interfaces were implementable as pinned. (batch_pipeline's
`run_once` returning `(success, message)` and disk's `stop() ->
(success, message, stats)` compose cleanly with the Trigger responses.)

## Round 1 (adversarial tester)

No previously-open bugs to verify (this is the first tester round). Coder
rebuttal/flag review:

- Coder-flagged `continuous_* data=false` unconditionally setting `Mode.OFF`
  (turning off a mode that was not active also stops the other): **accepted**
  as a literal reading of Sec 4 ("data=false: stop" with no which-mode
  qualifier). Left for the master tester per the coder's note; not filed as a
  bug.

Method: line-by-line walk of Sec 4 (QoS table, group table, all 13 service
semantics rows, race handling), Sec 5 propagation/ended behavior, Sec 8;
`py_compile`; then two in-container probe scripts
(`outputs/rosio_probe1.py`, `outputs/rosio_probe2.py`, run in
`deepstream-work:ds-ros` with ros2_ws mounted, no GPU) driving `DsRosNode`
with fake collaborators + real GrabState/Lifecycle. Verified executably:
exact QoS on all six publishers; all 13 services present with the exact Sec 4
group assignment (grp_capture Reentrant, others MutuallyExclusive); 1 Hz
timer in grp_fast; coalescing (2 racing mosaic callers -> exactly 1 publish,
identical stamps); re-arm-after-consumption publishes a fresh message;
ended-state fail-fast (<1 ms, message "source ended"); vlm rgb8
width/height/step/data bytes exact; snapshot writes PNG + .json sidecar and
answers the path; enqueue depth 1..3 / clear count; both toggles
(either-on-turns-other-off, worker set_continuous args); run_detect[_assess]
delegation incl. proceeding when ended; TargetBoxArray field mapping (seq
under lock, 640x368 q85 source_img decoded and size-checked, corner->center
bbox math, DETECTION_YOLO, gimbal w=1, copy-one-stamp-everywhere);
CasualtyImageCompressed (8 sorted clip_rgb_* annotations, probabilities,
bbox from the matching Detection, zeros when no det matches, all three
stamps equal); /ds/status keys/level; `_answer` exception containment;
record/stop message format. All conform.

Cross-module note: disk.md BUG disk-r1-1 (submit after DiskWorker.stop never
fires the Event) is *contained* on the ros_io side — `_snapshot` waits with
`SNAPSHOT_WRITE_TIMEOUT_S = 10.0`, so a grp_capture thread cannot hang
process shutdown; it answers `success=false "snapshot write timed out"`.

No blocker or major findings. Three minors:

### BUG rosio-r1-1: /ds/preview/compressed publisher QoS depth is 5, design says KEEP_LAST 1 [minor]

ros_io.py:164-165. The publisher uses `qos_profile_sensor_data`, which is
BEST_EFFORT / KEEP_LAST **5** / VOLATILE (verified in-container:
`sensor_data profile: BEST_EFFORT KEEP_LAST 5 VOLATILE`). DESIGN.md states
KEEP_LAST **1** for this topic in two places (Sec 4 topics table: "BEST_EFFORT,
KEEP_LAST 1 (sensor-data profile)"; Sec 8: "BEST_EFFORT/KEEP_LAST 1").
The design is self-contradictory — the ROS sensor-data profile it names in
the same cell has depth 5, and the pinned skeleton docstring says
"qos_profile_sensor_data" — so the coder followed one of the two readings.
Failure scenario: under congestion a subscriber can be delivered up to 5
buffered ~30 Hz frames (~170 ms stale) instead of only the newest; the
"a viewer that lags just misses frames" property is weakened, not broken.
Obvious fix: explicit `QoSProfile(reliability=BEST_EFFORT,
history=KEEP_LAST, depth=1, durability=VOLATILE)` satisfies both design
statements; alternatively document the depth-5 choice as the deliberate
resolution of the design's self-contradiction. Found by line-by-line QoS
conformance check + in-container profile dump.

### BUG rosio-r1-2: coalescing cache pins the last full-res frame per capture kind (~59 MB) indefinitely [minor]

ros_io.py:153-154, 234-241 (with frames.py:79-96). `_capture_results[kind] =
(waiter, message)` keeps the CaptureWaiter alive for coalesced-caller
identity checks, and the waiter's `_result` holds the full `CapturedFrame`
RGBA copy — 2560x1440x4 ≈ 14.75 MB. One frame per kind stays pinned until
the *next* successful capture of that kind replaces it, i.e. up to ~59 MB of
host RAM held forever after each capture kind has been used once (on top of
the design's budgeted 236 MB batch queue and the accepted ~11 MB /vlm_raw
latch). Demonstrated in `outputs/rosio_probe1.py`: after the capture
completed, `node._capture_results[MOSAIC][0]._result` is still the
CapturedFrame. Failure scenario: steady-state RSS ~59 MB higher than the
design's memory accounting; on a RAM-constrained host this margin was never
budgeted. Obvious fix: cache `(weakref.ref(waiter), message)` and compare
`cached[0]() is waiter` — every coalesced caller holds a strong ref to the
waiter while it can still hit the cache path, so the identity check stays
sound and the frame is freed as soon as the last caller returns. Found by
resource-leak walk of the coalescing bookkeeping; confirmed by probe.

### BUG rosio-r1-3: record/stop response silently discards the recorder's "finalize_pending" indicator [minor]

ros_io.py:454-462 (with disk.py:194). When the drain timed out AND the
force-finalize NULL is still wedged, `Recorder.stop` returns
`message="finalize_pending"` — but ros_io unconditionally rebuilds the
message from stats whenever `stats is not None`, so the service response
reads `path=... frames_written=... frames_dropped=... drained=false` and the
wedge indicator never reaches the caller (demonstrated with a fake recorder
in `outputs/rosio_probe1.py`; the only remaining trace is disk.py's stderr
WARNING). The formatted message conforms to the Sec 4 spec verbatim
("path, frames written, frames dropped, drained=true|false"), hence minor:
an operator can see the drain failed (`drained=false`) but cannot
distinguish "branch removed, file closed" from "filesink still wedged, NULL
pending" — and `record/start` will then fail with "previous recording still
finalizing" for no service-visible reason. Obvious fix: append
` finalize_pending=true` to the formatted message when the recorder's
message (or a dedicated stats field) says so. Found by tracing every value
Recorder.stop can return through the ros_io formatter.

Notes (not bugs):

- `publish_assessment` re-encodes the 640x368 source_img JPEG from raw RGBA
  once per assessed person plus once for the detections message (ros_bridge
  reused one already-encoded payload). ~5-10 ms per encode on the batch
  worker thread; ~1 s extra in a worst-case 16-frame multi-person assess run
  against the 30 s budget. Acceptable; flagging for awareness.
- A capture caller whose 2 s wait expires a moment before the frame arrives
  leaves the armed flag consumed with no publish (no caller left to produce).
  This matches "call blocks until published / failure implies no publish";
  recorded as the intended reading of Sec 4.
- grp_fast's sub-ms property can transiently stretch while the status timer
  blocks on `Recorder._lock` during a concurrent `record/start` `_attach`
  (element construction holds the lock, ~ms-scale). Not a deadlock; scope is
  disk.py's lock granularity, noted for the master tester.

## Master round 1

- **BUG MR1-ROSIO-1 (minor)** — Turning a continuous mode off leaves the
  auto-enqueued pending frames in the batch queue, and they silently join the
  next *manual* `run_detect`. Reproduction: enable `continuous_detect`, let
  it run, disable it; up to `continuous.run_size-1` frames remain queued
  (observed "cleared 3 frames" / "cleared 2 frames" from `/ds/batch/clear`
  immediately after disable, twice). Consequence observed in Sec 10 test 12:
  after 2 manual enqueues, `run_detect` reported "3/3 frames" and published a
  detection stamped ~3 minutes in the past (stamp 1784598146 vs now
  1784598314) — a stale frame from the earlier continuous session. Sec 4 says
  `data=false`: "stop" without specifying leftover semantics, so this is
  filed as a surprising-behavior minor, not a spec violation: either
  auto-clear (or final auto-run) on disable, or document that pending
  continuous frames persist and count toward the next manual run.

## Master round 2

- **BUG MR2-ROSIO-1 (minor, carry-over of MR1-ROSIO-1 — still open, no fix
  applied).** Disabling a continuous mode still leaves auto-enqueued frames
  in the pending deque; `_set_continuous(False, ...)` (ros_io.py:411-420)
  calls `grab.set_mode(Mode.OFF)` + `worker.set_continuous(False, ...)`
  (batch_pipeline.py:426-438) and neither clears nor final-runs the
  leftovers, and the README does not document the persistence. Reproduced
  this round three times: enable `continuous_detect`, sleep 2 s, disable,
  then `/ds/batch/clear` → "cleared 0 frames" / "cleared 2 frames" /
  "cleared 2 frames" (and "cleared 1 frames" after the test-13
  detect_assess session). Consequence unchanged from round 1: a later
  manual `run_detect` silently includes stale continuous-session frames
  with old stamps unless the caller clears first. Fix options unchanged:
  auto-clear (or final auto-run) on disable, or document the persistence in
  README/Sec 4.
- Everything else in ros_io re-verified end-to-end this round: one-shot
  latching + coalescing (tests 2-4), batch services incl. 6-frame run and
  assess gating with 8 clip_rgb_* annotations (5, 6), continuous ~10 Hz
  (7), ended-state fail-fast + idempotent stop (15), concurrent service
  independence — record/start and enqueue returned in 5 ms while a snapshot
  was in flight; capture/vlm concurrent (1.145 s) was faster than its solo
  baseline (~1.2 s, three runs), so no serialization behind the snapshot
  (14). Observation, not a bug: a `capture/vlm` call takes ~1.2 s
  wall (11 MB TRANSIENT_LOCAL publish) — within the 2 s design bound; and
  during vlm captures the preview worst 30-frame window dips to ~25 Hz
  (max inter-frame gap 98 ms, average 29.1 Hz) — within "~30 Hz".
