# batch_pipeline.py review log

## Interface notes

- `GrabState` exposes only `pending_depth()`, so the continuous-mode "oldest
  pending item > 200 ms" rule (Sec 4) cannot read the oldest `BatchItem`'s
  own stamp without a `swap_pending()` (which takes everything). The worker
  therefore tracks age from the first poll (50 ms interval) that observed a
  non-empty queue — correct within ~50 ms, verified by a unit-style check
  (trigger fired at 243 ms for a single queued frame). If exactness ever
  matters, a `GrabState.oldest_pending_ntp_ns()` accessor would make it
  precise; not requesting a change now since the approximation meets the
  design's intent (a bounded flush latency for partial batches).
- `Detection.object_id` / `FrameResult.assessments` keys: this pipeline has
  no tracker, so `obj_meta.object_id` is the untracked sentinel for every
  object (the existing repo falls back to a per-frame index via
  `get_detection_id`, but nothing in the batch pipeline ever stores an id in
  `misc_obj_info`). Both probes key objects by their ordinal position in the
  frame's `obj_meta_list` — the sgie appends tensor meta without reordering
  that list, so det/assess joins are stable and unique within a frame.
  Testers: `object_id` is per-frame, not cross-frame identity.

## Coder round 1

Implemented `batch_pipeline.py` in full:

- `build_batch_pipeline`: Sec 3.3 graph verbatim — appsrc `src_batch`
  (is-live, format=TIME, block=true, do-timestamp=false,
  max-bytes=268435456, RGBA WxH framerate=0/1 caps) -> `conv_batch`
  nvvideoconvert `output-buffers=16` -> `caps_batch` NVMM RGBA ->
  `mux_batch` (batch-size=`batch.engine_batch` (8), W/H,
  batched-push-timeout=100000, attach-sys-ts=false, live-source=false;
  linked via requested `sink_0`) -> `pgie_batch` -> `v_assess` valve
  drop=true -> `sgie_batch` (existing injury b8 config verbatim,
  process-mode=2, output-tensor-meta=true) -> `sink_batch` fakesink
  sync=false async=false. No state changes, no probes (per skeleton).
- `install_collect_probes`: `det_collect` on the pgie src pad (before the
  valve) collects per-frame `Detection` tuples keyed by
  `frame_meta.buf_pts`; `assess_collect` on the sgie src pad reuses
  `deepstream_yolo.assessment_runtime.parse_assessment_tensor_meta`,
  filtering tensor meta on `ASSESSMENT_GIE_ID` exactly like
  `assessment_runtime.assessment_probe` does. Both probes are
  exception-fenced and always return OK.
- `ResultCollector`: mutex+condition store; completion = every expected pts
  has detections, plus assessments when the run opened the valve (the sgie
  probe records empty dicts for person-free frames so completion still
  counts). Late results from a timed-out earlier run are dropped (counted),
  never joined. `results()` omits frames that never traversed the pgie
  (timeout) rather than fabricating empty detections — nvinfer emits a
  frame_meta per frame, so absence means loss, not zero objects.
- `BatchWorker`: `run_once` = reject (continuous active / empty queue) ->
  `swap_pending` snapshot -> valve set for the run -> push each item
  re-stamped with its feeder-assigned pts -> `wait_complete` (10 s cap,
  Sec 3.3) -> publish via injected callbacks -> valve restored drop=true.
  An internal `_run_lock` serializes `run_once` against continuous
  auto-runs (covers the toggle-off-mid-auto-run race), so the valve never
  toggles with buffers in flight. Continuous mode is the same
  `_execute_run` code path, triggered at `run_size` depth or the 200 ms
  age rule (see interface note). Failed pushes / collector timeouts return
  `success=false` with counts + reason in the message; collected frames are
  still published.

Validation:

- `python3 -m py_compile ds_ros_pipeline/batch_pipeline.py` clean.
- GPU-free logic run (host python, no Gst/pyds imports touched):
  collector join/order/stale-drop/assess-gating/timeout-omission; run_once
  against a real `GrabState` (enqueue -> on_frame -> run) publishing
  3/3 frames with valve log `[drop=False, drop=True]` for assess and
  `[drop=True, drop=True]` for detect-only; continuous mode rejecting
  manual runs, auto-running at depth 4, and flushing a single frame after
  243 ms via the age rule; clean start/stop join.
- In-container (`docker run --rm --gpus all deepstream-work:ds-ros`):
  `build_batch_pipeline(PipelineConfig(), 2560, 1440, ...)` + property
  assertions on every Sec 3.3 value (max-bytes, output-buffers=16,
  batch-size=8, batched-push-timeout=100000, attach-sys-ts=false,
  live-source=false, valve drop, sgie process-mode/output-tensor-meta,
  fakesink sync/async) + every src pad linked + `install_collect_probes`
  (exercises the pyds + assessment_runtime import path) -> BUILD_OK.
- Not exercised here (needs the full node): actual inference, buf_pts
  survival through appsrc->mux (design-asserted, Sec 5), engine build.
  That is master-tester territory (tests 5/6/7).

## Round 1

No previously-open bugs to verify (first adversarial round; the file above is
coder notes only).

Interface notes reviewed and ACCEPTED on evidence:

- 200 ms age-rule approximation: reproduced independently
  (`outputs/adv_bp_logic.py`) — a single queued frame auto-flushed at 241 ms,
  within the documented <=50 ms polling slack. Meets Sec 4's intent (bounded
  partial-batch flush latency).
- ordinal `object_id`: confirmed against the sgie config
  (`configs/generated/config_infer_secondary_injury_clip_vit_l14_336_b8.txt`:
  `operate-on-gie-id=1` matches the generated pgie's `gie-unique-id=1`,
  `classifier-async-mode=0` so meta is attached synchronously) and against
  `assessment_runtime.assessment_probe`'s identical iteration idiom — nvinfer
  neither adds, removes, nor reorders obj metas, so the pgie-src/sgie-src
  ordinal join is stable.

Verification performed (adversarial, both executed):

- `outputs/adv_bp_logic.py` (host, GPU-free, real GrabState/Lifecycle/config):
  collector completion order-independence (assess-before-det), stale-drop
  counting, timeout omission in `results()`, detect-only not waiting on
  assessments; run_once happy/empty/timeout/push-failure paths with valve
  logs `[open, close]` (assess) and `[closed, closed]` (detect); manual-run
  rejection while continuous; auto-run at run_size depth; age-rule flush;
  toggle-off-mid-auto-run then immediate manual run never interleaves
  (`_run_lock`); `stop()` joins cleanly. All pass.
- `outputs/adv_bp_gst.py` (in `deepstream-work:ds-ros`, GPU): independent
  readback of every Sec 3.3 property on the real built pipeline (all exact,
  incl. max-bytes=268435456, output-buffers=16, batched-push-timeout=100000,
  attach-sys-ts/live-source false, valve drop, sgie process-mode=2 +
  output-tensor-meta, fakesink sync/async false, every src pad linked);
  `install_collect_probes` import path OK; then the *production*
  `_push_items` (numpy -> tobytes -> new_wrapped -> pts) driven into the real
  appsrc -> nvvideoconvert(16) -> NVMM RGBA -> nvstreammux(b8) front section:
  16 frames pushed in 0.10 s, batches formed **[8, 8]** (the Sec 3.3
  pool-sizing regression is demonstrably fixed on this exact code path), and
  `frame_meta.buf_pts` == the pushed feeder pts, in order — the Sec 5/6
  join-key assertion the coder had left design-asserted is now measured.

### BUG BP-1: failed push still waits the full collector timeout [minor]

`batch_pipeline.py:496-497` — `_execute_run` sets
`pushed = self._push_items(items)` and then unconditionally calls
`wait_complete(wait_timeout)`. On a push failure (appsrc returns non-OK, e.g.
FLUSHING because the batch pipeline is shutting down while a continuous run
is in flight) the collector's `_expected` still contains every item's pts —
including frames that were never pushed — so the wait can only expire: up to
10 s of dead time, during which `BatchWorker.stop()` cannot join and a
`grp_batch` service caller sits blocked for nothing. Demonstrated in
`outputs/adv_bp_logic.py` (push_behavior="fail": 0-frame run still waited the
full 0.3 s test timeout; production value is 10 s). Low severity because the
documented shutdown order (ds_node docstring: workers stop before pipelines
go NULL) avoids the FLUSHING trigger in the normal path. Obvious fix: on
push failure, prune the expected set to the successfully pushed prefix (or
skip the wait when nothing was pushed).

### BUG BP-2: ResultCollector._stale is counted but unobservable [minor]

`batch_pipeline.py:92,107,118` — late results from a timed-out earlier run
are dropped and counted ("drop, count" per the coder notes), but `_stale` has
no accessor and is not folded into any /ds/status counter
(`ros_io.on_status_timer` publishes only `grab.counters()`). A recurring
collector timeout — the symptom the Sec 3.3 pool sizing exists to prevent —
would be invisible except via debugger. Obvious fix: expose it (e.g.
`ResultCollector.counters()`) and merge into the status values.

Verdict: no blocker or major bugs found. Sec 3.3 conformance is exact
(verified by property readback, not just code reading), the Sec 6 worker/
collector semantics hold under the race interleavings I could construct, and
the two empirical claims the design leans on (pool sizing, buf_pts joining)
reproduce on the production code path.
