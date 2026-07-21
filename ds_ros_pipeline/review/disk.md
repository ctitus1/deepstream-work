# disk.py

## Interface notes

- `write_png` gained two OPTIONAL keyword args, `pts=None, ntp_ns=None`
  (backward-compatible; `write_png(frame, path)` still works exactly as the
  skeleton pinned). Rationale: DESIGN.md Sec 5 says "snapshots get a `.json`
  sidecar with the same fields" — for *all* snapshots, but the pinned
  signature carried no stamp fields for the PNG variant. When both args are
  given, the same `.json` sidecar as the PPM path is written next to the PNG;
  ros_io may adopt it or ignore it. Every other public signature is unchanged.
- `Recorder.start` returns `(False, "previous recording still finalizing")`
  while a timed-out force-finalize NULL is still pending on its disposable
  thread (the hardening path). Rationale: the branch elements are still in
  the pipeline under their canonical Sec 3.1 names, so a re-attach would
  collide in `gst_bin_add`, and a wedged disk makes a new recording doomed
  anyway. The flag self-clears when the NULL completes.

## Coder round 1

Implemented all of disk.py per Sec 7 / Sec 3.1 Branch R / Sec 5:

- `DetachSequencer`: pure, fully injected. `stop(timeout)` installs the IDLE
  probe whose callback runs unlink -> send_eos -> release_pad (in that exact
  order, inside the callback body), then `wait_drain(timeout)`, then
  `finalize` unconditionally. Returns drained. No drain wait can precede
  unlink+release by construction (single code path).
- `Recorder`: attach follows Sec 7 Start order (request `t_ingest` pad via
  `request_pad_simple("src_%u")`, build elements, add, link chain, install
  probes, `sync_state_with_parent` downstream-first, link tee pad last).
  h265 branch: q_rec leaky=2/30 buffers/flush-on-eos=false, conv_rec,
  caps NVMM NV12, nvv4l2h265enc bitrate=config.record_bitrate control-rate=1
  iframeinterval=30 idrinterval=30, h265parse config-interval=-1, mpegtsmux,
  non-leaky q_disk 64MiB (buffers=0 time=0), filesink sync=false async=false.
  Raw variant: caps I420, q_disk 256MiB, `.yuv`. Probes: rec_sidecar
  (BUFFER, q_rec src pad — registry lookup by pts, `time.time_ns()` fallback
  so line count always matches file frames, appends `.jsonl`, counts
  frames_written); EOS probe (EVENT_DOWNSTREAM, sink_rec.sink) sets
  drain_done; q_rec `overrun` signal counts frames_dropped.
- Stop binds `DetachSequencer` to the real pad/element callables; the real
  IDLE probe wrapper returns `Gst.PadProbeReturn.REMOVE`. Finalize: drained
  => NULL+remove inline; not drained => NULL+remove on a disposable daemon
  thread with a second bounded `join(record.stop_timeout)` (accepted-minor
  hardening); if even that times out the stop response message is
  `"finalize_pending"` and a stderr warning is logged; stats still returned
  with drained=False.
- `on_source_eos` (loop=false): sets source_ended (start fails "source
  ended" forever after), and if recording, waits on the already-set
  drain_done, unlinks/releases the tee pad directly (no data flows
  post-EOS), NULLs+removes, stores stats; state 'finalized'; `stop()` is
  idempotent afterwards and returns those stats with success=True.
  stop/on_source_eos races are settled by a claim (`self._branch = None`
  under the mutex) — whoever claims the branch finalizes it, the other
  backs off.
- `DiskWorker`: single "disk" thread, FIFO `queue.Queue`, `submit` returns an
  Event set in `finally` (job exceptions logged to stderr, worker survives),
  `stop()` drains via sentinel then joins.
- Pure helpers: `record_path`/`snapshot_path` (UTC tag via
  `timestamps.utc_tag`), `append_sidecar_line` (single buffered append,
  `{"pts":…,"ntp_ns":…,"utc":"…"}`), `write_png` (cv2 RGBA->BGR, imwrite
  return checked), `write_ppm_with_sidecar` (P6 header + contiguous RGB +
  `.json` sidecar).
- Module import stays GStreamer-free (all `gi` imports deferred into
  methods), so tests.py can import `DetachSequencer` and the pure helpers
  anywhere.

Validation (all passing):

1. `python3 -m py_compile ds_ros_pipeline/disk.py` — clean.
2. Host pure tests (no Gst/numpy): DetachSequencer ordering
   `[install, unlink, eos, release, wait(5.0), finalize]`; timeout path
   returns drained=False with finalize still run and no wait before
   unlink+release; record/snapshot filename goldens incl. raw
   `rec_<tag>_2560x1440_I420.yuv` and its `.jsonl` sibling via
   `with_suffix`; sidecar JSONL line content round-trips through
   `json.loads`.
3. In `deepstream-work:ds-ros` (`docker run --rm -i --gpus all`):
   `import disk` clean; `write_ppm_with_sidecar` golden bytes
   (`P6\n6 4\n255\n` + 72 px bytes) + `.json` sidecar; `write_png` output
   re-read by cv2 with correct shape/channel values + sidecar; every Branch
   R element constructs and accepts its exact Sec 3.1 property values
   (nvv4l2h265enc bitrate/control-rate/iframeinterval/idrinterval, h265parse
   config-interval=-1, mpegtsmux, both queue configs, filesink sync/async);
   queue `overrun` signal connects; the `video/x-raw(memory:NVMM),format=NV12`
   caps string parses.

Not exercised here (needs the full running pipeline): live attach/detach
against a playing trunk — covered by Sec 10 tests 8-10/15/16 (master
tester).

## Round 1

No previously-open bugs to verify (first adversarial pass on this module).

Verification actually executed (adversarial tester, all evidence reproducible):

1. `python3 -m py_compile` clean; host import of disk.py is Gst-free as
   claimed (whole dependency chain config/source/live_pipeline/timestamps
   defers `gi`).
2. `outputs/probe_disk_round1.py` (host, pure logic): DetachSequencer inline
   AND deferred-callback ordering (`unlink -> eos -> release` strictly before
   the drain-wait return, finalize unconditional, timeout => drained=False);
   path/sidecar goldens (`.jsonl` parses via json.loads, raw
   `_WxH_I420.jsonl` sibling); Recorder state machine with faked Gst
   internals: stop happy path, finalize_pending refusal + self-clear, force-
   finalize bounded at ~stop_timeout with stderr warning, on_source_eos ->
   finalized -> idempotent stop -> `start` refused "source ended", and a
   200-trial concurrent stop-vs-on_source_eos race — the branch claim yields
   exactly one finalizer every time. DiskWorker FIFO order, exception
   survival, drain-then-join stop.
3. `outputs/probe_disk_gst.py` (deepstream-work:ds-ros, --gpus all, REAL
   attach/detach against a playing NVMM trunk: videotestsrc -> nvvideoconvert
   -> NVMM NV12 -> tee + leaky preview branch): h265 start/stop — drained=True
   in 0.05 s, 60 frames written == 60 sidecar lines, 1.2 MB .ts, branch fully
   removed (get_by_name None, tee pad count restored); **preview max
   inter-buffer gap across the stop instant = 34 ms** (the Sec 7/test-16
   observable — no trunk stall); raw variant file size an exact frame
   multiple (30.00 frames) == frames_written; a third start proves the
   canonical element names are reusable after detach.
4. `outputs/probe_disk_eos.py` (same container): a normal stop's branch-local
   EOS never surfaces as a pipeline bus EOS (0.8 s bus poll: none); the
   loop=false path — pipeline-wide EOS -> bus EOS -> on_source_eos() returns
   in 0.03 s, state finalized, branch removed, tee pad released, file valid,
   stop idempotent with stats == sidecar count, start refused "source ended".
5. `outputs/probe_disk_writers.py` (same container): write_png RGBA->BGR
   pixel order verified by cv2 re-read, `.json` sidecar exact, no sidecar
   when stamps absent; PPM `P6\n6 4\n255\n` golden header + RGB body order;
   **queue overrun-counts-drops assumption validated: 55 overrun signals ==
   55 actual leaky=2 drops** in a forced-starvation pipeline (silent
   defaults to false, checked via gst-inspect). Also gst-inspect confirmed
   nvv4l2h265enc bitrate/control-rate/iframeinterval/idrinterval exist and
   are NULL/READY-settable (disk.py sets them pre-sync — correct), h265parse
   config-interval doc allows -1, queue has flush-on-eos, mpegtsmux present.

Sec 7 / Sec 3.1 Branch R / Sec 5-disk walked requirement by requirement:
every element name, property value, probe type/pad, ordering, and pool/queue
size matches the design (q_rec leaky=2/30/0/0/flush-on-eos=false; caps NVMM
NV12 vs raw I420; enc CBR/iframeinterval=30/idrinterval=30; parse
config-interval=-1; q_disk leaky=0 64MiB/256MiB buffers=0 time=0; filesink
sync=false async=false; IDLE probe on the tee request pad; unlink -> EOS ->
release inside the callback; EOS probe on sink_rec.sink installed at attach;
async drain bounded by record.stop_timeout; force-finalize on timeout). No
silently skipped requirement found.

Notes (not bugs):
- Sec 5 says filenames embed "UTC ingest time of the first frame"; the code
  tags with wall time at attach. Forced by Sec 7 itself (filesink location
  must be set before any frame exists); skew <= ~1 frame interval + attach
  cost. Accepted as design-intent conformant.
- live_pipeline exposes `request_tee_pad`/`release_tee_pad` helpers whose
  docstring says disk.py uses them; disk.py calls the tee directly with
  identical semantics. Cosmetic drift only — either align disk.py to the
  helpers or drop them from live_pipeline's docstring.

### BUG disk-r1-1: DiskWorker.submit after stop() returns an Event that can never fire [minor]

disk.py:420-432. `stop()` enqueues the sentinel and joins; a `submit()` that
lands after the sentinel is never dequeued, so the returned Event is never
set. Demonstrated in `outputs/probe_disk_round1.py` ("submit after stop never
completes"). Failure scenario: ds_node.shutdown stops workers before the
executor (its documented order: "stop workers (batch, disk), executor.
shutdown"); a snapshot service call mid-flight then submits its write_png to
the stopped worker and its grp_capture thread waits forever — if ros_io waits
on the Event without a timeout, `executor.shutdown()` (no timeout) hangs the
whole process shutdown. Obvious fix inside disk.py: have `_run` drain-and-set
remaining job events after the sentinel, and make `submit` on a stopped/
never-started worker set the Event immediately (log to stderr) or raise.

### BUG disk-r1-2: _attach failure path leaks the requested tee pad and poisons retries (pipeline.add return unchecked) [minor]

disk.py:238-296. If anything after `request_pad_simple` raises (element
factory missing, `pipeline.add` refused, link failure), start() propagates
the exception with the tee request pad still allocated and any already-added
elements still in the pipeline. Because Branch R uses fixed canonical names,
the next start()'s `pipeline.add(element)` fails on the duplicate name —
return value unchecked (Gst.Bin.add returns False and only warns) — and the
subsequent `link` raises a misleading "failed to link" error; every retry
fails while orphan pads/elements accumulate. Only reachable when element
creation/adding/linking fails (broken install — the first attach would fail
too), hence minor. Obvious fix: wrap the body after the pad request in
try/except that releases the pad and NULLs+removes any added elements before
re-raising, and check `pipeline.add`'s return.

### BUG disk-r1-3: stop() during on_source_eos's finalize window returns "not recording" instead of the Sec 5 idempotent success [minor]

disk.py:184-216. `on_source_eos` claims the branch, then finalizes for a
measured ~30 ms (drain wait formality + pad release + NULL/remove) before
setting state "finalized". A `/ds/record/stop` arriving inside that window
sees state=="recording" with branch None and returns `(False, "not
recording", None)`, while Sec 5 promises stop after source EOS returns
success=true with the finalized stats. Found by interleaving analysis of the
claim protocol (the 200-trial race probe shows the claim itself is sound —
exactly one finalizer — this is only the transient reply). Realistic client
impact is negligible (requires calling stop within ~30 ms of the bus EOS);
obvious fix: in that branch of stop(), wait briefly for state to leave
"recording" (or return the in-progress path) instead of failing.
