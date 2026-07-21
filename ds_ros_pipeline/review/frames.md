## Coder round 1

Implemented `frames.py` completely within the pinned skeleton (no signature
changes):

- **`CaptureWaiter`**: `threading.Event` + result slot; `wait(timeout)` returns
  the `CapturedFrame` or None on timeout/ended. Internal `_fulfill` is called
  by GrabState outside its mutex.
- **`Lifecycle`**: one-way running->ended under a mutex; `guard` fails fast
  ("source ended") only for {capture, snapshot, enqueue, record_start};
  clear/mode/run/record_stop always pass (Sec 5).
- **`GrabState`**: single mutex over armed dict, enqueue counter, pending
  deque, mode/stride, drop counters. `on_frame` consumes ALL armed kinds and
  at most one enqueue decision under the mutex BEFORE copying; `copy()` is
  called exactly once per consuming frame, outside the mutex; every fulfilled
  waiter and the BatchItem share that one array. Coalescing per Sec 4: `arm`
  returns the existing waiter while a kind is armed; after consumption a new
  arm gets a fresh waiter (next frame). `request_enqueue` never coalesces; it
  reports depth = deque + not-yet-consumed pending requests (so N back-to-back
  calls answer 1..N, test 5) and refuses at `batch_capacity`. `swap_pending`
  snapshots-and-swaps the deque atomically under the mutex; `clear` empties
  the pending deque only. Continuous mode: stride counter resets on mode
  transition, fires on frames 0, stride, 2*stride, ...; a full deque counts
  `enqueue_drops` for manual decrements (still consumed) and
  `continuous_skips` for stride fires. `counters()` exposes
  {enqueue_drops, continuous_skips, copy_failures} for /ds/status.
- **Edge behavior chosen (not in the skeleton text)**: (a) `on_frame` with
  `ntp_ns=None` (registry rolled, no stream NTP) skips the frame entirely —
  armed flags stay armed, counter untouched; (b) if `copy()` raises, waiters
  are woken with None, the enqueue decrement stays consumed, and
  `copy_failures` increments — `on_frame` never raises; (c) at most ONE
  BatchItem per frame: a manual enqueue decrement takes precedence over a
  same-frame continuous-stride fire (avoids duplicate pts in the queue, which
  would break Sec 6 pts-joining); (d) `clear` deliberately does NOT reset the
  pending-enqueue counter (spec says it empties the pending deque only — an
  unconsumed enqueue still takes the next frame after a clear).
- **Gst/pyds glue**: `copy_surface` = `pyds.get_nvds_buf_surface(hash(buf),
  batch_id)` -> `np.array(copy=True, order="C")`; `make_grab_probe` extracts
  the first `NvDsFrameMeta` (mux_grab is batch-size=1) for `batch_id` and the
  resolve call, keys on `buffer.pts`, wraps the body in try/except and always
  returns `Gst.PadProbeReturn.OK`. All pyds/numpy/Gst imports deferred.

Validation:
- `python3 -m py_compile frames.py` clean.
- Pure-logic driver (fakes, no Gst) covering: coalesce/re-arm, all-four-kinds
  single-copy, enqueue counter N-distinct-frames + depth 1..N, capacity
  refusal + drops, snapshot-and-swap with mid-run enqueues surviving and
  clear-only-pending, continuous stride (0,3,6 of 7 frames) + mode mutual
  exclusion + zero copies when idle, lifecycle guard sets + notify_ended
  waking a blocked waiter with None + arm-after-ended resolving immediately,
  ntp_ns=None skip, waiter timeout, copy-failure path. All passed.
- In-container (`docker run --rm --gpus all deepstream-work:ds-ros`): module
  import, `make_grab_probe` construction under real Gst, and presence of the
  deferred pyds entry points — OK. Real surface mapping (risk 3) is left to
  the in-container test plan (test 2 canary).

Testers: tests.py stubs test_capture_arm_coalesce_and_rearm,
test_enqueue_counter_does_not_coalesce, test_snapshot_and_swap,
test_continuous_stride_auto_enqueue, test_ended_state_machine map 1:1 onto
the driver scenarios above; drive `on_frame(pts, ntp_ns, copy)` directly with
a counting `copy` callable.

## Interface notes

None — skeleton interfaces implemented as pinned. Added (non-breaking):
`GrabState.counters()` for the /ds/status drop counters the skeleton's thread
table already promised, `Lifecycle.RUNNING`/`ENDED` constants, and module
constant `ENDED_MESSAGE = "source ended"` for ros_io to reuse.

Tester round 1 addition: `arm()` returns the same `CaptureWaiter` to coalesced
callers but gives no discriminator for who publishes. Sec 4 / test 4 require
"exactly one topic message" per coalesced pair, and grp_capture is Reentrant,
so two service threads will both wake with the same CapturedFrame. ros_io can
implement publish-once without a signature change (first thread to enter a
per-`id(waiter)` dedup set under a ros_io lock publishes; others only format
the response), but this contract is currently written down nowhere. Either
document it in `arm()`'s docstring or give CaptureWaiter a
`claim_publish() -> bool` test-and-set. Flagged for the ros_io implementer;
not counted as an open frames.py bug.

## Round 1 (adversarial tester)

No previously-open tester bugs to verify (this is the first tester round; the
file contained only the coder's notes). The coder's pure-logic claims were
re-driven independently (`outputs/frames_race_probe.py` — coalesce/re-arm,
one-copy-per-frame, enqueue depths 1..N, swap/clear, copy-failure and
ntp=None paths all behave as documented) and the in-container import +
`make_grab_probe` construction claim was re-verified for real
(`outputs/grab_probe_e2e_check.py` runs the actual probe against a live
nvstreammux graph in `deepstream-work:ds-ros`). That same run demonstrates
BUG F1.

### BUG F1: grab probe keys the registry on post-mux buffer.pts, which nvstreammux regenerates — every frame is skipped [blocker]

- Where: `frames.py:385` (`pts = int(buffer.pts)` inside `make_grab_probe`),
  consumed at `frames.py:386-388`.
- Sec 5 states the rule explicitly: "In mux-bearing branches,
  `frame_meta.buf_pts` carries the same value" — Branch G is mux-bearing
  (`mux_grab`), and the registry is stamped on the trunk (pace src pad,
  pre-mux) with the feeder-assigned pts.
- Empirical: `outputs/mux_pts_check.py` / `mux_pts_check_offset.py`
  (deepstream-work:ds-ros, GPU, Branch-G topology: queue -> nvstreammux
  batch-size=1 attach-sys-ts=false live-source=false -> nvvideoconvert
  nvbuf-memory-type=3 -> RGBA capsfilter): with input pts starting at
  468 700 000 ns (the test clip's real first pts), the post-mux
  `buffer.pts` came out as 0, 33333333, 66666666, 99999999, ... — the mux
  REGENERATES output pts from zero (frame-count x truncated duration) and
  discards input pts entirely, while `frame_meta.buf_pts` preserved
  468700000, 502033333, ... exactly.
- Failure chain (all demonstrated end-to-end by
  `outputs/grab_probe_e2e_check.py`, which attaches the real
  `frames.make_grab_probe` on the post-mux capsfilter src pad with a real
  `TimestampRegistry` stamped pre-mux): `resolve()` tries stream helpers
  (`frame_timestamp` returns source "buf_pts"/"pts", neither in
  `WALL_CLOCK_TIMESTAMP_SOURCES = {"ntp", "ref"}`), falls back to
  `registry.get(mux-regenerated pts)` -> miss -> None -> `on_frame` skips the
  frame. Result over a 12-frame run: armed MOSAIC waiter never fulfilled
  (would burn the 2 s timeout on every `/ds/capture/*`, `/ds/snapshot*`
  call), `request_enqueue` never consumed (pending depth stayed 0). A
  control probe identical except `pts = frame_meta.buf_pts` fulfilled the
  capture (pts=468700000, real 240x320x4 uint8 copy) and enqueued 1 item.
  Every grab-branch service — captures, snapshots, enqueue, continuous
  modes — is nonfunctional as implemented.
- Secondary effect even if the registry somehow hit: `BatchItem.pts` would be
  the mux pts, not the feeder-assigned globally-unique pts Sec 5/6 require
  for result joining and cross-loop uniqueness (test 12).
- Fix (contained to `make_grab_probe`): after `_first_frame_meta`, use
  `pts = int(frame_meta.buf_pts) if frame_meta is not None else
  int(buffer.pts)`.

Coder round 2: fixed exactly as proposed — `make_grab_probe` now keys on
`frame_meta.buf_pts` (fallback `buffer.pts` only when no frame meta). Verified
by re-running your `outputs/grab_probe_e2e_check.py` unmodified in
`deepstream-work:ds-ros` (GPU): the as-implemented probe now reports
FULFILLED / pending depth 1, matching the buf_pts control
(pts=468700000, 240x320x4 uint8 copy).

### BUG F2: copy_surface never unmaps the surface, deviating from the design-cited motion-branch pattern [minor]

- Where: `frames.py:349-350`.
- Sec 3.1 justifies the grab branch as "same dGPU-mappable-surface trick as
  the proven motion branch"; that proven code
  (`src/deepstream_yolo/motion.py:741-760`) pairs
  `pyds.get_nvds_buf_surface` with `pyds.unmap_nvds_buf_surface` in a
  guarded epilogue. `copy_surface` maps and never unmaps.
- Checked pyds 1.2.0 docstrings in-container: unmap is mandatory only on
  Jetson ("For Jetson, a matching call ... must be made"); on x86_64 unified
  memory it is a no-op-if-unmapped safeguard, so there is no leak on this
  deployment — hence minor, not major. Matching the precedent
  (try/finally `pyds.unmap_nvds_buf_surface(hash(gst_buffer), batch_id)`,
  exception-swallowed like motion.py) costs nothing and keeps the module
  Jetson-portable.

Coder round 2: `copy_surface` now wraps the np.array copy in try/finally with
`pyds.unmap_nvds_buf_surface(hash(gst_buffer), batch_id)` in the finally,
exception-swallowed, matching motion.py. Exercised on real surfaces in the
same `grab_probe_e2e_check.py` run (both probes' copies go through
`frames.copy_surface`); copies still come out correct.

### BUG F3: enqueue accepted-then-silently-dropped during the copy window [minor]

- Where: `frames.py:202-207` (decision + counter decrement, lock #1) vs
  `frames.py:230-236` (append, lock #2); `request_enqueue` depth math at
  `frames.py:268`.
- Between the two lock sections the frame being copied (a few ms,
  Sec 6) is counted in neither `_pending_enqueues` nor `_pending`, so a
  concurrent `/ds/batch/enqueue` at the capacity boundary computes depth one
  low, is accepted (`success=true`, depth reported), and its frame is later
  dropped at consumption (`enqueue_drops`).
- Demonstrated deterministically in `outputs/frames_race_probe.py`
  ("over-admission window"): capacity=1, a `request_enqueue` issued from
  inside the copy callable -> two accepts, one landed, one dropped.
- Realistic path: hammering enqueue near capacity (test 5 does 6
  back-to-back calls, though not at the cap). Self-reporting via the drop
  counter, hence minor. Fix idea: an `_in_flight` count incremented with the
  decision under lock #1, decremented at append/drop under lock #2, and
  included in `request_enqueue`'s depth sum.

Coder round 2: implemented your `_in_flight` idea exactly: incremented under
lock #1 whenever an enqueue decision is made (manual or stride fire),
decremented under lock #2 at append/drop AND in the copy-failure path;
`request_enqueue` depth is now `len(pending) + pending_enqueues + in_flight`.
Re-drove your capacity=1 request-from-inside-copy scenario: the mid-copy
request is now refused `(False, 1)`, exactly one item lands, zero drops; a
copy failure releases the slot (follow-up request accepted at depth 1).

### BUG F4: arm() vs notify_ended() race strands a waiter for the full 2 s timeout [minor]

- Where: `frames.py:247-257` — `arm` reads `self._lifecycle.state` BEFORE
  taking the GrabState lock; `notify_ended` (`frames.py:325-335`) runs once,
  from the bus-EOS handler.
- Interleaving: arm sees RUNNING; `mark_ended` + `notify_ended` complete
  (armed dict empty, nothing to wake); arm then inserts its waiter. No
  future frame will ever arrive and no second wakeup exists, so the caller
  waits out the full CAPTURE_TIMEOUT_S — violating Sec 5's "fail fast the
  same way (no 2 s timeout)" for post-ended captures. Caller still gets
  success=false, and the window is microseconds once per process, hence
  minor.
- Demonstrated with a controlled interleaving in
  `outputs/frames_race_probe.py` ("arm-vs-ended stranded waiter").
- Fix: re-check lifecycle after inserting the waiter (fulfill None and
  remove if ended), or keep an `_ended` flag inside GrabState set under its
  own mutex by `notify_ended` and checked by `arm` under the same lock.

Coder round 2: took your second option — `notify_ended` sets
`GrabState._ended` under the GrabState lock; `arm` re-checks it under the
same lock before inserting and returns an already-resolved-None waiter if
set (the pre-lock lifecycle fast-path stays for the common case). Verified
with a deterministic interleaving (a lifecycle whose `state` getter triggers
`mark_ended` + `notify_ended` right after arm's fast-path read): the waiter
resolves immediately with None instead of stranding for the timeout.

### BUG F5: continuous-stride clock pauses on manual-enqueue frames; ntp=None skip is silent [minor]

- Where: `frames.py:202-215` — the stride branch is an `elif`, so a frame
  consumed by a manual enqueue neither fires nor advances `_stride_count`.
  The at-most-one-BatchItem-per-frame choice itself is sound (duplicate pts
  would break Sec 6 joins), but freezing the stride clock shifts the cadence:
  demonstrated in `outputs/frames_race_probe.py` — with stride 3 and one
  manual request interleaved, enqueued frames were 0, 3, 4, 7 (two adjacent
  fires, then a stretched gap) instead of an "every continuous.stride-th
  frame" cadence (Sec 4). Fix: advance `_stride_count` on every frame while
  mode is on, regardless of who consumed the frame.
- Same function: the `ntp_ns is None` early-return (`frames.py:191-195`)
  skips silently, but `timestamps.resolve`'s contract says "caller logs and
  skips the frame". Add a rate-limited/logged notice — a persistent
  resolve-miss (exactly what F1 produces) is otherwise invisible except as
  capture timeouts.

Coder round 2: (a) stride clock decoupled from consumption: `_stride_count`
now advances on every frame while mode is on; a manual decrement still takes
precedence on a fire frame (at most one BatchItem per frame, no
`continuous_skips` counted for the superseded fire since the deque wasn't
full). Re-drove your stride-3 + one-manual scenario: enqueued pts are now
0,2(manual),3,6,9 — fire cadence 0,3,6,9 unshifted; manual landing ON a fire
frame yields 0,3,6,9 with one item at pts 3. (b) `ntp_ns is None` skips now
log a rate-limited warning (first skip, then every 100th) and count into a
new `resolve_skips` key in `counters()` for /ds/status visibility.

Round-1 verdict: 1 blocker (F1), 4 minors (F2-F5) open. The pure state
machine (GrabState/Lifecycle/CaptureWaiter) is solid — every Sec 4 race rule
(coalescing, exactly-one-consumption-per-arming, non-coalescing counter,
snapshot-and-swap, clear-pending-only) checked out under direct adversarial
driving; the defects are concentrated in the Gst/pyds adapter layer.

## Coder round 2

All five round-1 bugs fixed (no rebuttals); per-bug details are annotated
under each entry above. Summary of code changes in `frames.py`:

- `make_grab_probe`: registry/BatchItem key is `frame_meta.buf_pts`
  (fallback `buffer.pts` only if frame meta is absent) — F1.
- `copy_surface`: try/finally `pyds.unmap_nvds_buf_surface`, exception
  swallowed, matching `src/deepstream_yolo/motion.py` — F2.
- `GrabState`: new `_in_flight` counter closes the copy-window over-admission
  (incremented with the enqueue decision, decremented at append/drop and on
  copy failure, included in `request_enqueue` depth) — F3.
- `GrabState._ended` flag set by `notify_ended` under the GrabState lock;
  `arm` re-checks it under that lock so a racing arm resolves immediately
  with None — F4.
- `on_frame`: stride clock advances every frame while mode is on (manual
  enqueue no longer freezes cadence; still at most one BatchItem/frame);
  `ntp_ns=None` skips log rate-limited (1st, every 100th) and are counted in
  a new non-breaking `counters()["resolve_skips"]` — F5.

No signature changes; skeleton interfaces untouched. No new interface notes.

Validation:
- `python3 -m py_compile frames.py` clean.
- Pure-logic driver (14 checks): F5 cadence with manual interleave
  (0,3,6,9 fires preserved; manual-on-fire-frame dedupes), resolve_skips
  counting, F3 mid-copy refusal at capacity=1 + slot release on copy
  failure, F4 deterministic arm-vs-notify_ended interleaving, plus
  regressions (coalesce/re-arm, one-copy-per-frame, depths 1..N + refusal).
  All pass.
- In-container (`docker run --rm --gpus all deepstream-work:ds-ros`): the
  tester's own `outputs/grab_probe_e2e_check.py`, unmodified — the
  as-implemented probe now FULFILLS the MOSAIC waiter and enqueues 1 item,
  identical to the buf_pts control (pts=468700000, 240x320x4 uint8), with
  the unmapping `copy_surface` on the real surface path.

## Round 2 (adversarial tester)

Verification of the round-1 bugs (all re-checked against the code and
re-executed, not just accepted from the coder's notes):

- BUG F1: FIXED (verified round 2). `frames.py:427-431` now keys on
  `frame_meta.buf_pts` (fallback `buffer.pts` only when frame meta is
  absent). Re-ran `outputs/grab_probe_e2e_check.py` UNMODIFIED in
  `deepstream-work:ds-ros` (GPU): the as-implemented probe reports
  FULFILLED / pending depth 1, identical to the buf_pts control
  (pts=468700000, 240x320x4 uint8 copy).
- BUG F2: FIXED (verified round 2). `frames.py:386-393`: try/finally with
  `pyds.unmap_nvds_buf_surface(hash(gst_buffer), batch_id)`, exception
  swallowed — matches the motion.py precedent. Exercised on real surfaces in
  the same e2e rerun (both probes route through `frames.copy_surface`);
  copies correct, no errors.
- BUG F3: FIXED (verified round 2). `_in_flight` incremented with the
  enqueue decision (`frames.py:234-237`), decremented at append
  (`frames.py:254-256`) and in the copy-failure path (`frames.py:243-246`),
  included in `request_enqueue` depth (`frames.py:300-303`). Re-drove
  capacity=1 with a `request_enqueue` issued from inside the copy callable
  (`outputs/frames_race_probe_r2.py`): mid-copy request now refused
  `(False, 1)`, exactly one item lands, zero drops; a copy failure releases
  the slot (follow-up accepted at depth 1).
- BUG F4: FIXED (verified round 2). `notify_ended` sets `GrabState._ended`
  under the GrabState lock (`frames.py:367-368`); `arm` re-checks it under
  that lock (`frames.py:277-282`). Re-drove the deterministic interleaving
  (lifecycle whose `state` getter parks arm after its fast-path read while
  `mark_ended` + `notify_ended` complete): the waiter now resolves None in
  <1 ms instead of stranding for the timeout.
- BUG F5: FIXED (verified round 2). (a) Stride clock advances on every frame
  while mode is on (`frames.py:217-221`); re-drove stride 3 with a manual
  request interleaved: enqueued pts 0,2(manual),3,6,9 — fire cadence
  unshifted; manual landing ON a fire frame yields 0,3,6 with one item at
  pts 3 (no duplicate pts). (b) `ntp_ns=None` skips log rate-limited
  (1st/every 100th, `frames.py:199-206`) and count into
  `counters()["resolve_skips"]` — both observed in the driver run.

New findings: none. Beyond re-verification, the module was re-attacked along
fresh lines, all clean:

- Re-walked Sec 4 requirement by requirement (arming = mutex-guarded flag +
  shared waiter; consume-before-copy under the mutex; coalesce-before-frame /
  re-arm-after-consumption; non-coalescing counter with N-distinct-frame
  semantics; swap-under-mutex with mid-run enqueues surviving;
  clear-pending-only) and Sec 6 (copy path = get_nvds_buf_surface ->
  np.array(copy=True) + unmap, capped deque, BatchItem carries
  {frame, ntp_ns, feeder pts}, copies only for signalled/strided frames,
  at-most-one-item-per-frame keeps pts unique for the Sec 6 join). No
  deviation found. Sec 5 ended behavior (guard fail-fast set, notify wake
  with None, arm-after-ended immediate None) exact.
- `outputs/frames_race_probe_r2.py` (22 checks, all PASS): the F3/F4/F5
  re-drives above, plus new interleavings — swap_pending called from INSIDE
  the copy callable (in-flight item correctly lands in the fresh deque, i.e.
  survives the run per Sec 4); clear from inside the copy (see note below);
  an `_in_flight` leak hunt across every on_frame path (manual append,
  manual+waiter, stride fire, copy failure, ntp=None skip — admission stays
  exact at capacity afterwards); accepted-but-unconsumed request counted
  across a swap; idempotent `set_mode(DETECT)` twice does NOT reset the
  stride phase while a real transition does; refusal reports current depth;
  200 iterations of concurrent `on_frame` vs `notify_ended` on one armed
  waiter — exactly one fulfill, result is either the frame or None, never a
  hang.
- `outputs/frames_stress_r2.py`: 4 s hammer — one full-speed on_frame thread
  vs four service threads (request_enqueue / arm+wait / swap_pending / mode
  toggling + counters). Result: 3036 items landed, 0 duplicate pts, max
  observed depth == capacity (never exceeded), 0 starved captures, 0
  exceptions out of on_frame, 0 accounting errors.
- Lock audit: no nested locks anywhere in frames.py (arm reads
  `lifecycle.state` before taking the GrabState lock; notify_ended and every
  GrabState method take only the GrabState lock; fulfills and copies run
  outside it) — no deadlock or ordering hazard possible within this module.

Notes (not bugs, recorded for the record):

- A frame whose copy is in flight when `clear()` runs lands in the deque
  just after the clear (demonstrated in the r2 probe). This is the same
  class as the round-1-accepted "an unconsumed enqueue still takes the next
  frame after a clear" (edge (d)): the in-flight frame is not in the pending
  deque at clear time, so Sec 4's "clear empties only the current (pending)
  deque" is satisfied literally. Window is the few-ms copy; not counted as a
  bug.
- The append-time capacity recheck in `on_frame` (`frames.py:257-261`,
  `_enqueue_drops` on a full deque at append) is now effectively dead code:
  on_frame is the only appender and runs on a single streaming thread, and
  admission (`request_enqueue` + `_in_flight`) is strict, so the deque
  cannot fill between decision and append. Harmless defensive redundancy;
  left as is.
- tests.py's five frames-related stubs are still `NotImplementedError` —
  that file has its own owner per the manifest; flagged there, not counted
  against frames.py.
- The round-1 interface note (coalesced-waiter publish-once discriminator
  for ros_io) stands unchanged; still not a frames.py bug.

Round-2 verdict: 0 blockers, 0 majors, 0 minors open. frames.py is clean.
