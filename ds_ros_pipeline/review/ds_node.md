# ds_node.py review file

## Interface notes

- No sibling interface changes were needed; every pinned signature
  (`App.__init__/build/run/on_bus_message/shutdown`, `main`) is implemented
  as declared.
- Parameter resolution uses a short-lived bootstrap `Node("ds_pipeline")`
  (created, `declare_parameters`, destroyed) because `DsRosNode` takes the
  finished collaborators which are themselves built FROM the config — a
  chicken-and-egg the pinned constructor order forces. After the real
  `DsRosNode` exists, `config.declare_parameters` is called on it a second
  time purely so `ros2 param list/get` works; `has_parameter` guards make
  the re-declaration a no-op value-wise (verified in-container: overrides
  land on the bootstrap AND read back identically on the second node).

## Coder round 1

Implemented `ds_node.py` completely per DESIGN.md Sec 1/2/5/9 plus the
`--prebuild` path in `run.sh` (Sec 11 risk 10):

- **build()** (main thread, exact skeleton order): `Gst.init`; `rclpy.init`
  with `SignalHandlerOptions.NO` (falls back to plain init if the enum is
  missing) so GLib owns SIGINT/SIGTERM; bootstrap-node config resolution
  (see Interface notes); `infer_configs.write_batch_yolo_config(
  cfg.batch_engine_batch)` + `sgie_config_path()`; registry/lifecycle/grab;
  `source.create_source`; `live_pipeline.build_live_pipeline` +
  `install_ingest_stamp_probe` + `frames.make_grab_probe` installed as a
  BUFFER probe on `caps_grab`'s src pad; `batch_pipeline.build` +
  `install_collect_probes` + `BatchWorker` (publish callbacks are lambdas
  closing over `self._node`, assigned before any run can execute);
  `disk.Recorder` + `DiskWorker`; `ros_io.DsRosNode` with
  `loop_count_fn=lambda: source.n_loops`; `connect_preview`; bus watches
  (`bus.add_watch`) on BOTH pipelines.
- **run()**: `GLib.unix_signal_add` for SIGINT+SIGTERM (handler quits the
  MainLoop; stays installed); batch then live to PLAYING (only
  `FAILURE` is fatal — live returns NO_PREROLL by design); disk worker,
  batch worker, `source.start()` (feeder); `MultiThreadedExecutor(
  num_threads=8)` spun on daemon thread "ros"; `GLib.MainLoop.run()` on the
  main thread; `shutdown()` in a finally.
- **on_bus_message()**: origin discrimination by bus identity (pygobject
  wrapper caching verified in-container — see validation). ERROR: loud
  stderr log with debug string, exit_code=1, quit loop. WARNING: logged.
  Live-pipeline EOS (Sec 5, loop=false only): `lifecycle.mark_ended()` →
  `grab.notify_ended()` → `recorder.on_source_eos()`, in that order (frames'
  documented ordering), and does NOT quit — /ds/status keeps publishing at
  1 Hz until SIGINT. Batch-pipeline EOS: logged as unexpected.
- **shutdown()** (idempotent, None-safe at every step so a partial build
  still tears down): finalize an active recording via the NORMAL
  `recorder.stop()` path first (needs the still-PLAYING graph for the
  Sec 7 drain) → batch worker stop → `source.stop()` (joins feeder) →
  `executor.shutdown(timeout 10 s)` + join "ros" thread (in-flight
  service callbacks get to finish; disk worker still alive for their
  snapshot writes) → disk worker stop (drains queued jobs) → remove bus
  watches + both pipelines NULL → `destroy_node` + `rclpy.shutdown`.
- **--prebuild** (`run.sh --prebuild` or `ds_node.py --prebuild`): writes
  the b{N} pgie config, asserts the sgie config, then cycles the batch
  pipeline through PLAYING (10 min state timeout) so nvinfer builds/caches
  the engine, then NULL and exit 0/1 with bus errors printed. run.sh
  dispatches it BEFORE the ROS sourcing/CDCL_ROS_SETUP check (prebuild
  needs no ROS — usable in a bare container/image-build warmup). Normal
  startup path in run.sh unchanged.

Validation run:

1. `python3 -m py_compile ds_ros_pipeline/ds_node.py` — clean; `bash -n
   run.sh` — clean.
2. Host (no ROS/Gst env): `import ds_node`, `App([...])` construction,
   double `shutdown()` on an empty App (idempotence + None-safety) — OK,
   no PyGIWarning (all Gst imports go through `gi.require_version` on every
   entry path including teardown/prebuild).
3. In `deepstream-work:ds-ros` (repo + ros2_ws mounted, ROS env sourced):
   `rclpy.signals.SignalHandlerOptions` and `MultiThreadedExecutor` exist;
   `GLib.unix_signal_add` exists; full `import ds_node` + `import ros_io`
   chain (cdcl_umd_msgs etc.) succeeds; the bootstrap param path honors
   `--ros-args -p source.loop:=false -p batch.capacity:=7` and the
   re-declaration on a second same-name node returns identical values.
4. In-container Gst check: `pipeline.get_bus()` returns a cached wrapper
   and the `add_watch` callback receives that same object (`is` and `==`)
   — the live/batch origin test in `on_bus_message` is sound.

Not validated here (needs GPU/compose, master tester): PLAYING state
changes, EOS→ended end-to-end (Sec 10 test 15), engine warmup timing.

## Round 1

Adversarial tester, round 1. No previously-open bugs existed (first tester
pass). Coder validation claims re-checked: items 1-4 of the coder's
validation list are accurate as far as they go (re-confirmed: py_compile
clean; `SignalHandlerOptions`/`MultiThreadedExecutor(num_threads=8)` /
`Executor.shutdown(timeout_sec)` exist in-container; `bus.add_watch(
GLib.PRIORITY_DEFAULT, cb)` arity correct; `get_bus()` wrapper caching holds
and the watch callback receives the identical wrapper, so the live/batch
origin test is sound; bootstrap+re-declare round-trips `--ros-args -p`
overrides). But the validation never executed the *interleaving* build()
actually performs (`import rclpy` -> `Gst.init` -> `rclpy.init` -> `Node`),
and that exact interleaving crashes the process — see DSN-1.

Also verified conforming (no findings): EOS bus handling order
mark_ended -> notify_ended -> on_source_eos with no loop-quit, batch-EOS
warning path, ERROR -> exit_code=1 + quit (driven with mock collaborators
in-container: `outputs/dsnode_eos_probe.py`); GLib unix-signal ->
loop.quit -> executor shutdown/join/destroy sequence works end-to-end and a
signal delivered *before* loop.run() (the engine-build window, where
set_state blocks 1-3 min) is deferred, not lost (`outputs/
dsnode_signal_bisect.py`); shutdown() is idempotent and None-safe on a
partial build; only StateChangeReturn.FAILURE is treated fatal (live
pipeline legitimately returns NO_PREROLL); all 13 sibling-interface
signatures ds_node consumes match their definitions; the 14 unit tests in
tests.py pass in-container.

### BUG DSN-1: startup crash — `import rclpy` before `Gst.init(None)` makes the bootstrap `Node()` abort [blocker] — FIXED (verified round 2)

`ds_ros_pipeline/ds_node.py:88` (`import rclpy`), `:90` (`Gst.init(None)`),
`:98-101` (`rclpy.init`), crash at `:109` (`bootstrap = Node("ds_pipeline")`).

**The process cannot start at all in the deployment container.** Running the
real entrypoint (`python3 ds_ros_pipeline/ds_node.py`) in
`deepstream-work:ds-ros` — including with compose-matching `--ipc=host
--network=host` — dies at ds_node.py:109 on every run (SIGABRT, occasionally
SIGSEGV; exit 134/139), before any pipeline or service exists:

```
Fatal Python error: Aborted
  File ".../rclpy/node.py", line 175 in __init__     # _rclpy.Node(...)
  File ".../ds_ros_pipeline/ds_node.py", line 109 in build
```

Mechanism (gdb backtrace, `outputs/`): `Gst.init(None)` dlopens
`libunwind.so.8` into the process (verified via /proc/self/maps: absent
before Gst.init, present after). When `libfastrtps` was loaded *earlier*
(by `import rclpy` at ds_node.py:88), a routine internally-caught C++
exception in Fast DDS SHM-transport setup
(`SharedMemTransport::CreateInputChannelResource`) then unwinds through
libunwind's incompatible `_Unwind_Resume` -> `std::terminate` -> abort at
the first `Node()` creation. The GPU-less sandbox is not the trigger: the
DS plugins that fail to load play no role, and the crash reproduces with
`ipc: host`/host networking exactly as compose runs it.

Minimal-pair matrix (`outputs/dsnode_order_matrix.py`, 2-4 runs each, 100%
reproducible both ways):

| Order | Result |
|---|---|
| V1 `import rclpy` -> `Gst.init` -> `rclpy.init` -> `Node` (= build() today) | **abort** |
| V2 `Gst.init` -> `import rclpy` -> `rclpy.init` -> `Node` | OK |
| V3 `import rclpy` -> `rclpy.init` -> `Node` -> `Gst.init` | OK |
| V4 `rclpy.init` -> `Node` -> `Gst.init` -> second `Node` | OK |

Fix (either, both empirically validated): (a) **move `rclpy.init` (and
optionally the bootstrap-node parameter resolution) BEFORE `Gst.init(None)`**
— V4 proves nodes created after Gst.init are fine once rclpy.init preceded
it, so DsRosNode later in build() is unaffected; or (b) move the
`import rclpy` statement below `Gst.init(None)` (V2). Option (a) as a full
build()-prefix reorder was validated end-to-end 3/3 in-container with the
real cdcl_umd_msgs workspace mounted — bootstrap params, Gst.init, ros_io
import, second `Node("ds_pipeline")`, MultiThreadedExecutor spin, GLib
SIGINT shutdown all clean (`outputs/dsnode_fixorder_probe.py`). No other
module imports rclpy before build() reaches this point (ros_io is imported
later at :148), and the `--prebuild` path never touches rclpy, so the fix
is contained to build()'s first ~25 lines.

How found: attempted to validate the coder's "bootstrap param path" claim
with Gst initialized in the same process (as build() actually runs), got a
segfault, bisected the import/init interleaving, confirmed through the real
entrypoint, then gdb + /proc/self/maps for the mechanism.

**Coder round 2:** Fixed with option (a): `rclpy.init` + the bootstrap
`Node("ds_pipeline")` param resolution now run BEFORE `Gst.init(None)` /
`GLib.MainLoop()` in build() (ds_node.py:86-118), with a comment pinning the
ordering to this bug. Verified via the real entrypoint in
`deepstream-work:ds-ros` (`--ipc=host --network=host`, ros2_ws mounted,
run.sh-identical env): 3/3 runs get past the bootstrap Node with no
SIGABRT/SIGSEGV and proceed to the expected GPU-less failure
(`RuntimeError: Missing GStreamer element: nvv4l2decoder` from
source.create_source, exit 1 — the point your matrix predicted).

### BUG DSN-2: unguarded `on_bus_message` — one exception silently kills the bus watch [minor] — FIXED (verified round 2)

`ds_ros_pipeline/ds_node.py:212-248`. Demonstrated in-container
(`outputs/dsnode_api_probe.py`): when a PyGObject bus-watch callback raises,
the traceback is printed and the callback's return becomes falsy — GLib
**removes the watch permanently** (a second posted message was never
delivered). `on_bus_message` has no try/except, and its live-EOS path calls
`recorder.on_source_eos()`, which does real Gst work (pad unlink/release,
element NULL, pipeline.remove) inside the callback. If any of that — or a
`print` to a broken stdout — raises once, the process permanently loses
ERROR handling for that pipeline (a later fatal pipeline error would no
longer quit the loop; an EOS transition could be half-applied) with nothing
but a one-time traceback to show for it. Obvious fix: wrap the handler body
in try/except (log, fall through) so it always returns True.

**Coder round 2:** Fixed: the body moved to `_handle_bus_message`;
`on_bus_message` is now `try: self._handle_bus_message(...) except:
traceback.print_exc()` and unconditionally `return True`. Verified
in-container on the real GLib machinery (`outputs/dsnode_round2_fix_probe.py`):
with a lifecycle collaborator that raises on `mark_ended`, the watch callback
returns True, the traceback is logged, and a bus watch installed via
`bus.add_watch` still receives 2/2 posted EOS messages (watch not removed).

### BUG DSN-3: shutdown can hang on `worker.stop()`'s unbounded join — batch pipeline is NULLed only after [minor] — FIXED (verified round 2)

`ds_ros_pipeline/ds_node.py:258-259` (`self.worker.stop()`) vs `:275`
(`self._teardown_gst()`); cross-file with
`ds_ros_pipeline/batch_pipeline.py:396-404` (`stop` joins with no timeout).
A blocked `appsrc push-buffer` (`block=true`) in a mid-flight run is
unblocked only by the batch pipeline leaving PLAYING — which shutdown()
performs *after* the unbounded join, so a wedged batch pipeline turns
shutdown into a permanent hang (SIGINT/SIGTERM do nothing at that point:
the loop has already quit and the GLib handlers only quit the loop). With
default config this is unreachable — a full swap is at most
`batch.capacity=16` frames = 236 MB, which fits appsrc's 256 MiB queue, so
pushes never block and the join is bounded by the 10 s collector timeout —
but any `batch.capacity` >= 19 (a plain `-p` override away) makes the 19th
push block forever against a wedged pipeline. Also contradicts the file's
own `SHUTDOWN_JOIN_TIMEOUT_S` comment ("bound every shutdown wait", :40).
Fix: NULL (or flush) the batch pipeline/appsrc before joining the worker,
or bound the join and NULL on timeout. (DiskWorker.stop's unbounded join
at :274 has the same shape but only local file writes behind it.)

**Coder round 2:** Fixed with your second option (bound the join, NULL on
timeout), implemented entirely in ds_node.py (`_stop_batch_worker`) since
`BatchWorker.stop()`'s signature is pinned: `worker.stop()` runs on a
daemon helper thread, joined with `SHUTDOWN_JOIN_TIMEOUT_S`; on timeout the
batch pipeline is forced to NULL (releasing a blocked `push-buffer`) and
the join retried with `SHUTDOWN_JOIN_TIMEOUT_S + 1` (covers the residual
10 s collect wait; shutdown then proceeds regardless, with a warning). The
normal path is unchanged — a prompt stop never touches the pipeline, and
`_teardown_gst`'s later NULL is idempotent. Verified in-container
(`outputs/dsnode_round2_fix_probe.py`, with the timeout shrunk to 0.5 s):
a worker wedged on an Event that only the pipeline's NULL releases is
unblocked and shutdown completes in 0.50 s; the fast path leaves
`set_state` uncalled. DiskWorker.stop's join left as-is per your parenthetical
(local file writes only).

### BUG DSN-4: `--prebuild` ignores parameter overrides — warms the wrong engine for non-default `batch.engine_batch` [minor] — FIXED (verified round 2)

`ds_ros_pipeline/ds_node.py:334-335` (`PipelineConfig()` defaults),
`ds_ros_pipeline/run.sh:13-16` (drops every arg after `--prebuild`).
Sec 11 risk 1 names `batch.engine_batch=4` as the documented VRAM fallback;
a deployment running with that override still gets the b8 engine warmed by
prebuild (1-3 min spent, plus VRAM for a build that then goes unused) and
pays the b4 build again at first real start. Design only says "consider a
--prebuild", so defaults-only is defensible — but plumbing an optional
`--engine-batch N` (or honoring `--ros-args` via the bootstrap-node path,
post-DSN-1-fix) would make the warmup actually match the deployment.

**Coder round 2:** Fixed via the `--engine-batch N` option (kept the
no-ROS property of prebuild rather than the `--ros-args` route):
`prebuild(engine_batch: int | None)` builds
`PipelineConfig(batch_engine_batch=N)` when given; `main` parses
`--prebuild [--engine-batch N]` (malformed/missing value -> usage + exit 2);
run.sh now `shift`s and forwards the remaining args to the `--prebuild`
exec. Verified in-container (`outputs/dsnode_round2_fix_probe.py`):
`main(["--prebuild","--engine-batch","4"])` reaches `prebuild(4)`, bare
`--prebuild` passes None (defaults), bad/missing values exit 2, and
`prebuild(engine_batch=4)` writes the b4 nvinfer config and hands a config
with `batch_engine_batch=4` to `build_batch_pipeline` (pipeline stubbed —
no engine build in the sandbox). `bash -n run.sh` clean.

Probe scripts kept under `outputs/` (dsnode_order_matrix.py,
dsnode_exact_order_probe.py, dsnode_fixorder_probe.py, dsnode_api_probe.py,
dsnode_eos_probe.py, dsnode_rclpy_probe.py, dsnode_signal_probe.py,
dsnode_signal_bisect.py, dsnode_crash_pinpoint.py) for the fix round to
re-run: after the DSN-1 fix, `dsnode_order_matrix.py` V1 becoming moot and
`timeout 60 python3 ds_ros_pipeline/ds_node.py` in the container must get
past line 109 (it will then fail at nvv4l2decoder creation without a GPU —
that part is the master tester's).

## Coder round 2

Addressed all four Round 1 bugs (details under each bug entry above):

- **DSN-1 (blocker):** build() reordered — `rclpy.init` + bootstrap-node
  parameter resolution now precede `Gst.init(None)` (your validated option
  (a)); comment in build() documents why the order is load-bearing.
- **DSN-2:** `on_bus_message` wraps the real handler
  (`_handle_bus_message`) in try/except and always returns True, so a
  raising delivery can never remove the bus watch.
- **DSN-3:** new `_stop_batch_worker`: bounded join of `worker.stop()` via
  a helper thread; on timeout the batch pipeline is NULLed to release a
  blocked appsrc push, then the join is retried once and shutdown proceeds.
  No change to batch_pipeline.py (pinned interface).
- **DSN-4:** `--prebuild [--engine-batch N]` warms the b{N} engine; run.sh
  forwards post-`--prebuild` args.

Validation (all in `deepstream-work:ds-ros` unless noted):

1. `python3 -m py_compile ds_ros_pipeline/ds_node.py` and `bash -n
   ds_ros_pipeline/run.sh` — clean (host).
2. Real entrypoint, compose-matching env (`--ipc=host --network=host`,
   ros2_ws mounted, ROS + overlay sourced, run.sh's exact PYTHONPATH):
   3/3 runs pass the former DSN-1 abort point and exit 1 at the expected
   GPU-less `nvv4l2decoder` RuntimeError — no SIGABRT/SIGSEGV (exit 134/139
   gone).
3. `outputs/dsnode_round2_fix_probe.py` — 13/13 PASS covering DSN-2 (real
   `bus.add_watch` survives a raising delivery, 2/2 messages), DSN-3
   (wedged worker released by pipeline NULL, bounded at the configured
   timeout; fast path touches nothing), DSN-4 (arg parsing incl. error
   exits; b4 config written and b4 config object reaches
   build_batch_pipeline).
4. `python3 ds_ros_pipeline/tests.py` — 15 tests, OK (no regression).

Still master-tester territory: PLAYING with a GPU, EOS→ended end-to-end
(Sec 10 test 15), real engine warmup timing, and a real wedged-batch
shutdown under load.

## Round 2

Adversarial tester, round 2. Every previously-open bug was re-verified
against the actual code and behavior in `deepstream-work:ds-ros`:

- **DSN-1 FIXED (verified round 2)**: re-ran the REAL entrypoint
  (`ds_ros_pipeline/run.sh` via `bash -lc`, compose-matching `--ipc=host
  --network=host`, ros2_ws mounted ro, `CDCL_ROS_SETUP` set) 3/3 times: no
  SIGABRT/SIGSEGV; every run passes the former abort point (bootstrap
  `Node("ds_pipeline")`, now at ds_node.py:111, BEFORE `Gst.init` at :118)
  and exits 1 at the predicted GPU-less `RuntimeError: Missing GStreamer
  element: nvv4l2decoder` (source.py:347). `outputs/dsnode_fixorder_probe.py`
  (the exact fixed sequence incl. ros_io/cdcl_umd_msgs import + second Node +
  executor spin + SIGINT shutdown after Gst.init) re-run: all 6 steps clean.
- **DSN-2 FIXED (verified round 2)**: `on_bus_message` (ds_node.py:227-231)
  wraps `_handle_bus_message` and unconditionally returns True.
  `outputs/dsnode_round2_fix_probe.py`: raising lifecycle collaborator ->
  return True + traceback logged; on a real `bus.add_watch` a raising
  delivery does NOT remove the watch (2/2 posted EOS delivered). Re-ran
  `outputs/dsnode_eos_probe.py` against the restructured handler: live-EOS
  order mark_ended -> notify_ended -> on_source_eos intact, batch-EOS
  warning-only, ERROR -> exit_code=1 + loop_quit, all paths return True.
- **DSN-3 FIXED (verified round 2)**: `_stop_batch_worker`
  (ds_node.py:290-322) joins `worker.stop()` on a daemon helper thread with
  `SHUTDOWN_JOIN_TIMEOUT_S`; on timeout NULLs the batch pipeline and retries
  with timeout+1, then proceeds. Probe: a stop() wedged on an Event released
  only by pipeline NULL completes in 0.50 s at a 0.5 s timeout; the fast path
  never touches the pipeline. My fresh `outputs/
  dsnode_round2_shutdown_probe.py` additionally confirms the wedged path
  composes with the rest of shutdown (double batch-NULL is exercised and
  harmless; teardown runs to completion).
- **DSN-4 FIXED (verified round 2)**: `--prebuild [--engine-batch N]`
  (ds_node.py:369-444, run.sh:13-17). Probe: parse paths incl. exit-2 error
  cases; `prebuild(4)` writes the b4 config and hands
  `batch_engine_batch=4` to `build_batch_pipeline`. Also verified END TO END
  through run.sh in the bare container: `ds_ros_pipeline/run.sh --prebuild
  --engine-batch 4` forwards the args, writes
  `generated/ds_ros_infer_batch_yolo12x_640_640x384_b4.txt`, and proceeds to
  the batch-pipeline build (dies only at the sandbox's missing
  `nvvideoconvert`, exit 1 with traceback — correct for a GPU-less host).

No coder rebuttals to adjudicate. Additional round-2 verification, all clean:

- `python3 -m py_compile ds_node.py`, `bash -n run.sh` — clean;
  `python3 ds_ros_pipeline/tests.py` in-container — 15 tests OK.
- Full `shutdown()` ordering driven with mock collaborators
  (`outputs/dsnode_round2_shutdown_probe.py`, 10/10 PASS): recorder.stop
  (recording active, stats logged) -> worker.stop -> source.stop ->
  executor.shutdown(10.0) -> disk.stop -> remove_watch x2 -> live NULL ->
  batch NULL -> destroy_node; second shutdown() is a strict no-op; run()'s
  batch-FAILURE path returns 1, never starts live/workers, and still tears
  down in order.
- Design walk (Sec 1/2/5 + Sec 9 row + Sec 4 thread budget): thread names
  and owners match Sec 2 exactly (main=GLib.MainLoop + both bus watches,
  "feeder"/"ros"(8)/"batch"/"disk"); ingest_stamp probe on pace src pad and
  grab probe (BUFFER) on caps_grab src pad match the Sec 3.1 graph
  annotations; Sec 5 ended transition matches word for word (no quit, status
  keeps publishing, recorder finalized via on_source_eos); all consumed
  sibling signatures re-checked against their definitions (incl.
  resolve_fn(registry, pts, frame_meta, buffer), LiveParts fields,
  DsRosNode ctor, DiskWorker/BatchWorker/SourceBin not-started-safe stop()).
- Observation, not a bug: the "ros" executor thread runs `executor.spin`
  unwrapped, so an exception escaping a callback would kill services
  silently — but all 13 services route through ros_io._answer (catches
  everything) and the 1 Hz status-timer body only touches locked state and
  a valid publisher, so no realistic raise path exists today.

### BUG DSN-5: `record/start` accepted during the shutdown window attaches a branch that is never finalized [minor]

`ds_ros_pipeline/ds_node.py:270` (`_finalize_recording`) vs `:276`
(`executor.shutdown`). Between finalizing the active recording (step 1 of
shutdown()) and stopping the executor (step 4), service callbacks keep being
served by design — and that window is long: `_stop_batch_worker` alone may
take up to ~21 s in the wedged case, plus `source.stop()`. A
`/ds/record/start` call landing in it finds `recorder.state == "idle"` and a
still-PLAYING live graph, so it attaches a fresh Branch R and returns
success; nothing ever stops it, and `_teardown_gst` (:287) then hard-NULLs
the live pipeline with the branch mid-write — no EOS through mpegtsmux, no
drain, no stats, an unfinalized `.ts`/`.raw` left on disk while the client
was told recording started. Found by walking every service against each
shutdown step; not demonstrated live (needs GPU + a client racing SIGINT).
Obvious fix candidates: quiesce first (move executor shutdown+join ahead of
`_finalize_recording` — the graph is still PLAYING until `_teardown_gst`,
which is all recorder.stop() needs, and Recorder's internal lock already
serializes a concurrent in-flight record/stop), or have shutdown() set a
flag that `_answer`-guarded mutating services check. Kept minor: the window
requires a client actively issuing record/start during SIGINT teardown, and
the stray TS file is truncated-but-playable, not corrupting.

Round-2 verdict: DSN-1..4 all fixed and verified; one new minor (DSN-5),
zero blocker/major open. Probe scripts for this round:
`outputs/dsnode_round2_fix_probe.py` (coder's, re-run),
`outputs/dsnode_round2_shutdown_probe.py` (new),
`outputs/dsnode_eos_probe.py` / `dsnode_fixorder_probe.py` (round 1, re-run).
Still master-tester territory: PLAYING with a GPU, EOS->ended end-to-end
(Sec 10 test 15), real engine warmup, wedged-batch shutdown under load.
