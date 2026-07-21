## Coder round 1

Implemented tests.py in full: all 13 skeleton tests plus one extra
(`test_force_finalize_disposable_thread`, required by the Sec 10 seam list's
"disposable-thread finalize path"). Structural change from the skeleton: the
module-level pytest functions became methods (same names, same docstrings) on
unittest.TestCase classes so the file runs BOTH ways — `python3 -m pytest
ds_ros_pipeline/tests.py` and `python3 ds_ros_pipeline/tests.py`
(unittest.main). No other module imports tests.py, so nothing else is affected.

Coverage notes for testers:
- Timestamps: registry roundtrip, default-2048 + small-capacity eviction
  order; resolve order exercised by assigning fake helper pairs to
  `timestamps._stream_helpers` (stream-NTP wins, non-wall-clock/None fall back,
  helpers never consulted without meta, None when both miss). Helpers restored
  via addCleanup.
- Source: schedule_pts strict monotonicity + global uniqueness over 5 loops on
  a 287-AU clip-like list and a variable-duration list; compute_loop_span on
  uniform/single/decode-reordered lists + empty ValueError; select_variant for
  file/FILE/rtsp/rtsps/unknown/plain-path, plus register_variant("Mock") ->
  create_source dispatches to the mock factory with (uri, config, None) and
  never touches Gst (cleanup pops the registration).
- frames.GrabState: coalesce-to-one-waiter/one-copy, re-arm after consumption,
  all armed kinds served from one frame with ONE copy, idle frames copy
  nothing, ntp=None skip keeps arming and bumps resolve_skips (asserted via
  assertLogs); enqueue counter 1..N depths, capacity False at depth cap
  (pending + queued counted), clear resets; snapshot-and-swap with mid-run
  enqueue surviving and clear touching only pending; continuous stride fires
  frames 0/3/6, mode switch returns previous + resets stride clock, same-mode
  set does not reset, manual enqueue on a stride-fire frame yields exactly one
  item, OFF stops, full-deque skip counts continuous_skips.
- Lifecycle: fail-fast set vs always-allowed set before/after mark_ended
  (idempotent), notify_ended wakes armed waiters with None, post-ended arm
  returns an already-resolved None waiter.
- disk.DetachSequencer: exact log ["install_idle_probe","unlink","send_eos",
  "release_pad",("wait_drain",t),"finalize"] in both drained outcomes (probe
  callback invoked inline per Sec 10); timeout path uses a never-set
  threading.Event as wait_drain — drained=False, bounded elapsed, finalize
  last. Recorder._finalize_branch tested with `_null_and_remove` stubbed on the
  instance: drained => inline on calling thread; not drained => runs on the
  "rec-force-finalize" thread and _finalize_pending clears.
- infer_configs: hermetic golden — synthetic b1 template (byte-identical shape
  to the repo's real one) in a temp repo root; render(8) == literal golden
  (batch-size=8 + _b8_gpu0_fp16.engine, everything else verbatim); render(1)
  round-trips to the template byte-for-byte; ValueError on batch<1;
  FileNotFoundError on missing template; b8 output filename check; plus a
  guarded cross-check against the real repo template (only the two keys
  differ).

No bugs found in the modules under test; no interface notes — all seams were
importable without Gst/pyds/rclpy exactly as their docstrings promise.

Validation: `python3 -m py_compile` clean; `python3 ds_ros_pipeline/tests.py`
=> 14/14 OK on the host (no pytest installed there); `docker run --rm -v
/home/user/deepstream-work:/workspace/deepstream-work deepstream-work:ds-ros
python3 -m pytest .../tests.py -q` => 14 passed (pytest 6.2.5); a sys.modules
scan after a full in-process run confirms gi/pyds/rclpy are never imported.

## Round 1

No previously-open bugs to verify (this is the first tester round; the file
above is the coder's report only).

Method: line-by-line walk of the Sec 10 "GPU/ROS-free seams" paragraph against
tests.py, re-ran the suite both ways (host `python3 ds_ros_pipeline/tests.py`
=> 14/14 OK; `docker run --rm ... deepstream-work:ds-ros python3 -m pytest
.../tests.py -q` => 14 passed), re-verified the sys.modules purity claim
(no gi/pyds/rclpy — nor numpy/cv2 — after a full in-process run), and
mutation-tested the suite: 14 targeted mutations of the modules under test
applied to a throwaway copy in /tmp/mutant, checking each is caught. Killed
(suite fails as it should): detach wait_drain-before-probe-install, detach
skip-finalize-when-not-drained, detach release-before-EOS, finalize
always-inline (no disposable thread), lifecycle enqueue-not-fail-fast,
resolve registry-preferred-over-stream-NTP, registry evict-newest,
loop_span last-pushed-instead-of-max-pts, grab copy-per-waiter-kind, arm
non-coalescing, ntp-None-consumes-arming, infer engine-suffix-hardcoded-b8,
same-mode-set-resets-stride. All seven seam bullets of Sec 10 are present and
non-vacuous except as noted below; coverage-vs-spec is otherwise complete
(registry bounding + REGISTRY_CAPACITY==2048 pinned, resolve order, pts
monotonicity/uniqueness across wraps, loop_span derivation incl. decode
reorder + empty ValueError, arm/coalesce/counter/swap, ended fail-fast sets
matching Sec 5 verbatim, exact detach ordering in both drain outcomes,
drain_done-never-fires timeout, disposable-thread finalize, golden b8 +
b1 round-trip + real-template cross-check, variant selection + mock factory
dispatch with cleanup). Three mutations survived — findings:

### BUG T1: stride-clock-reset assertion is vacuous (9 pre-switch frames == a
multiple of stride 3) [major]

tests.py:338-348 (`test_continuous_stride_auto_enqueue`). The test drives 9
frames in Mode.DETECT (stride 3), then switches to DETECT_ASSESS and asserts
"resets the stride clock: the next frame fires immediately". But after 9
frames `_stride_count == 9` and `9 % 3 == 0`, so frame 2000 fires *with or
without* the reset — the assertion cannot fail for the behavior it names.
Demonstrated: deleting `self._stride_count = 0` from `frames.set_mode` in a
/tmp copy leaves all 14 tests green, while a probe script switching after 8
frames shows the mutant observably diverges (correct impl fires the first
post-switch frame, mutant fires nothing). The vacuity cascades: in the same
mutant, frame 3000 in the "manual enqueue on a stride-fire frame" block is
*not* a stride-fire frame (13 % 3 == 1), so the at-most-one-item-per-frame
coalescing claim at tests.py:358-364 is also no longer exercised. frames.py's
documented contract ("The stride counter resets on transition",
frames.py:318) and the coder's own coverage note claim this is pinned; it is
not. Fix (one character): drive 8 frames instead of 9 at tests.py:338
(`for index in range(8)`) — the expected fire list [1000, 1003, 1006] is
unchanged, and the post-switch immediate fire then discriminates reset vs
no-reset (8 % 3 != 0). Found by mutation testing.

Coder round 2: applied the suggested fix — drive 8 frames (with a comment
pinning why 8 must not be a multiple of the stride); fire list and all later
assertions unchanged. Re-applied the delete-`_stride_count = 0` mutant to a
/tmp copy: now killed by test_continuous_stride_auto_enqueue.

### BUG T2: request_enqueue's in-flight (mid-copy) depth accounting is
untested [minor]

tests.py:268-303 (`test_enqueue_counter_does_not_coalesce`).
frames.request_enqueue's docstring promises "Frames mid-copy (consumed but
not yet appended) count too, so the copy window cannot over-admit at the
capacity boundary" (frames.py:296-298, backed by the `_in_flight` counter).
No test ever observes depth while a copy is in flight — all on_frame calls
are synchronous — so deleting `+ self._in_flight` from the depth sum leaves
all 14 tests green (demonstrated in /tmp/mutant). Regression impact: a
service-thread request_enqueue racing the streaming-thread copy at the
capacity boundary returns success for a frame that then lands as an
enqueue_drop. Testable at the pure seam: pass a copy callable that itself
calls `state.request_enqueue()` (reentrant, outside the mutex by design) with
the deque at capacity-1 and assert it returns (False, capacity). Found by
mutation testing.

Coder round 2: added `test_enqueue_depth_counts_in_flight_copy`
(TestGrabState): batch_capacity=2, one frame queued, then a request_enqueue
issued from inside the copy callable while the second frame is in flight —
asserts it returns (False, 2), enqueue_drops stays 0, and swap_pending yields
exactly [100, 133]. Re-applied the drop-`+ self._in_flight` mutant to a /tmp
copy: now killed by the new test.

### BUG T3: schedule_pts linear formula not pinned — a drifting loop_span
survives monotonicity/uniqueness [minor]

tests.py:142-161 (`test_pts_schedule_monotonic_unique_across_wraps`). The
Sec 10 seam text's required assertions (global monotonicity + uniqueness
across wraps, loop_span derivation) are all present, but the exact Sec 5
formula `buf.pts = au.pts + n_loops * loop_span` is never asserted directly:
mutating schedule_pts to `au_pts + n_loops * (loop_span - 1)` stays strictly
monotonic and unique (the wrap gap just shrinks by 1 ns/loop) and passes all
14 tests. Only an exact-collision error (e.g. span short by one full frame
duration) is caught. A drifting schedule silently erodes the gapless
30.000 fps wrap cadence Sec 5 measured. One-line fix: assert
`schedule_pts(au.pts, n, span) == au.pts + n * span` inside the existing
loop, or assert the cross-wrap delta equals the last AU's duration for the
uniform list. Found by mutation testing.

Coder round 2: did both suggested fixes in
test_pts_schedule_monotonic_unique_across_wraps — the schedule is now built
in an explicit loop asserting the exact Sec 5 formula per value, and a new
block asserts every consecutive delta of the uniform 287-AU schedule
(including all four wrap boundaries) is exactly 33_333_333 ns (gapless
cadence). Re-applied the `loop_span - 1` drift mutant to a /tmp copy: now
killed by this test.

Non-findings (checked, fine): suite runs both ways and cwd-independent
(sys.path pinned from __file__); TestTimestamps saves/restores
`_stream_helpers` via addCleanup (exception-safe); mock variant registration
popped via addCleanup; no repo-level conftest.py/pytest.ini and no
first-party pytest-collectable basename collision with ds_ros_pipeline/
tests.py (only site-packages inside .venv-* match); timeout assertions are
generous (elapsed < 2.0 for a 0.05 s wait) — no flake risk spotted; disk.py
import chain (via live_pipeline) verified Gst-free at module top; fail-fast/
always-allowed service-name sets match frames._FAIL_FAST and DESIGN Sec 5
verbatim; the guarded real-repo-template cross-check actually executes on
this checkout (template present) and passes.

## Coder round 2

All three Round 1 bugs addressed (notes under each entry above); no rebuttals
— every finding was reproducible and correct. Changes, all in tests.py:

- T1: test_continuous_stride_auto_enqueue drives 8 pre-switch frames instead
  of 9 (8 % 3 != 0), with a comment explaining the constraint so a future
  edit does not regress it to a stride multiple. Assertions unchanged.
- T2: new test test_enqueue_depth_counts_in_flight_copy pins the
  `+ _in_flight` term of request_enqueue's depth: a reentrant
  request_enqueue issued from inside the copy callable (deque at
  capacity-1, one frame mid-copy) must return (False, capacity), with
  enqueue_drops == 0 and no extra BatchItem. Suite is now 15 tests.
- T3: test_pts_schedule_monotonic_unique_across_wraps asserts the exact
  Sec 5 linear formula for every scheduled value on both AU lists, plus a
  gapless-cadence check: all consecutive deltas of the uniform 287-AU
  schedule across 5 loops (wrap boundaries included) equal 33_333_333 ns.

No module code touched; no interface notes.

Validation: python3 -m py_compile clean; host `python3
ds_ros_pipeline/tests.py` => 15/15 OK; `docker run --rm -v
/home/user/deepstream-work:/workspace/deepstream-work deepstream-work:ds-ros
python3 -m pytest .../tests.py -q` => 15 passed. Re-ran the three surviving
Round 1 mutants against throwaway /tmp copies: stride-reset deletion killed
by test_continuous_stride_auto_enqueue, in-flight-term deletion killed by
test_enqueue_depth_counts_in_flight_copy, loop_span-1 drift killed by
test_pts_schedule_monotonic_unique_across_wraps (each with exactly the one
expected failure, 14 others green).

## Round 2

Verification of Round 1 bugs (each mutant independently re-applied to a fresh
/tmp/mutant copy of the package, suite re-run):

- BUG T1: FIXED (verified round 2). tests.py:387 drives `range(8)` with the
  pinning comment; the delete-`_stride_count = 0` mutant now fails
  test_continuous_stride_auto_enqueue (expected [2000], got []).
- BUG T2: FIXED (verified round 2). test_enqueue_depth_counts_in_flight_copy
  (tests.py:323-349) is present and correct — the reentrant request lands
  while `_in_flight == 1` with one frame queued at batch_capacity=2, outside
  the mutex (no deadlock; frames.on_frame calls copy() unlocked). The
  drop-`+ self._in_flight` mutant now fails exactly that test ((True, 3) vs
  expected (False, 2) — observed as mid_copy mismatch).
- BUG T3: FIXED (verified round 2). The exact-formula assertion
  (tests.py:157-162) and the gapless-cadence delta-set check
  (tests.py:170-179, including all four wrap boundaries) are both present;
  the `loop_span - 1` drift mutant now fails with "pts schedule deviates
  from the Sec 5 formula".

Re-validation: host `python3 ds_ros_pipeline/tests.py` => 15/15 OK, stable
over 30 consecutive runs (no flakes in the threaded detach/finalize tests);
`docker run --rm ... deepstream-work:ds-ros python3 -m pytest .../tests.py
-q` => 15 passed; in-process sys.modules scan after a full run: no gi, pyds,
rclpy, numpy, or cv2 imported; the guarded real-repo-template cross-check in
test_infer_config_golden executes on this checkout (template present).
Sec 10 seam walk (all seven bullets) re-done against the current file: every
bullet present and non-vacuous. Fresh mutation pass over the modules under
test found three surviving mutants — findings below, all coverage gaps (no
test in the file asserts anything false; the modules themselves conform).

### BUG T4: compute_loop_span's min-pts derivation unpinned — decode-order
test only pins the max side [minor]

tests.py:194-200 (`test_loop_span_derivation`, "Decode order != presentation
order" block). The reordered list's first-pushed AU (pts 0) is also the
min-pts AU, so mutating source.compute_loop_span:72 from
`first_pts = min(au.pts for au in aus)` to `first_pts = aus[0].pts` leaves
all 15 tests green (demonstrated in /tmp/mutant) — the comment claims the
span "still derives from the max-pts AU, not the last-pushed one" but only
the max side discriminates. The min side is load-bearing for the design's
own clip: an open-GOP HEVC stream decodes CRA-first with the RASL leading
pictures (lower pts) after it (DESIGN Sec 5 / Sec 11 risk 11), i.e.
decode-first != presentation-first exactly on the first_pts side; risk 11's
suggested wrap fix explicitly invites future edits to this span logic. A
regressed aus[0]-based span is short by the CRA/RASL offset, so
schedule_pts collides/regresses at every wrap — a runtime blocker the suite
would not catch. Fix: reorder the synthetic list so the first AU is not the
min (e.g. pts 66, 0, 100, 33 — expected span unchanged: 100 - 0 + 33).
Found by mutation testing.

### BUG T5: on_frame's copy-failure path entirely untested (in_flight leak,
waiter fulfillment, copy_failures counter all unpinned) [minor]

tests.py (TestGrabState — no test passes a raising copy callable).
frames.on_frame:242-250 documents "Never blocks, never raises" and on a
copy() exception must (a) decrement `_in_flight` for an admitted enqueue,
(b) fulfill armed waiters with None, (c) count `copy_failures`. None is
pinned: both the remove-`self._in_flight -= 1`-from-except mutant and the
remove-`waiter._fulfill(None)` mutant leave all 15 tests green
(demonstrated in /tmp/mutant). Regression impact is real: a leaked
_in_flight permanently inflates request_enqueue's depth, so after
batch_capacity transient copy failures the enqueue service returns
(False, …) forever; unfulfilled waiters turn a fast failure into a full
capture timeout. Fully testable at the pure seam: arm a waiter, issue a
request_enqueue, call on_frame with a copy callable that raises — assert
the waiter resolves to None, counters()["copy_failures"] == 1, and (with
batch_capacity=1) a subsequent request_enqueue still returns (True, 1).
Found by mutation testing.

### BUG T6: arm's under-lock ended re-check (notify_ended race window)
unpinned [minor]

tests.py:431-464 (`test_ended_state_machine`) reaches post-ended arm only
via lifecycle.mark_ended() + notify_ended(), which short-circuits at the
first check (frames.arm:272); deleting the documented race guard at
frames.arm:277-282 (`if self._ended:` under the lock) leaves all 15 tests
green (demonstrated in /tmp/mutant). The guarded window is real per the
code's own comment — a waiter inserted after notify_ended's sweep is never
woken, so the racing capture call hangs its full timeout instead of failing
fast ("no 2 s hang", Sec 5 / Sec 10 test 15). Deterministically testable at
the pure seam: call state.notify_ended() and then state.arm(...) while the
Lifecycle still reads RUNNING — the returned waiter must already be
resolved to None. Found by mutation testing.

Non-findings (checked, fine): the manual-enqueue-on-full-deque drop branch
(frames.on_frame:225-226) is unreachable under the documented single-probe
threading model (request_enqueue's depth check, including _in_flight, cannot
over-admit) — its lack of coverage is not a gap; the success-path
`_in_flight -= 1` (frames.on_frame:256) IS pinned (mutant killed by
test_enqueue_counter_does_not_coalesce's post-clear (True, 1)); the
force-finalize test's immediate assertions are race-free because
Recorder._finalize_branch joins the disposable thread (bounded by
record_stop_timeout=2.0 with an instant stub); DetachSequencer mutants
(finalize-before-wait, reordered callback body, unconditional drained=True)
are all killed by the full-log equality assertions; resolve()'s
`frame_meta is not None or buffer is not None` gate is pinned (an `and`
mutant fails the 777 assertion); DiskWorker and the utc_tag/record_path/
sidecar helpers are outside the Sec 10 seam list and rightly untested here.
