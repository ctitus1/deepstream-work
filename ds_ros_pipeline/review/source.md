# source.py review log

## Coder round 1

Implemented `source.py` completely per DESIGN.md Sec 3.1/5/9 and risks 5/6/11.

What was built:

- Pure seams (no Gst at import): `compute_loop_span` (max_pts − min_pts +
  duration of the max-pts AU — equals the docstring's last−first+duration for
  in-order lists and is robust to reorder), `schedule_pts`
  (`au.pts + n_loops*loop_span`), `select_variant` (file/rtsp(s)/registered
  scheme, else ValueError), `register_variant` (module-level registry; the
  future-IPC slot of Sec 11 risk 5; factory called as
  `factory(uri, config, None)` from `create_source`).
- `extract_access_units`: filesrc ! qtdemux ! {h265parse|h264parse}
  config-interval=-1 ! byte-stream/au capsfilter ! appsink sync=false; parser
  chosen from the qtdemux pad caps in a pad-added callback (h264 files work).
  Pull loop via `try-pull-sample` with bus-error checks, a 30 s stall
  deadline, and the `source.max_preload_mb` guard (RuntimeError suggesting
  loop=false, Sec 11 risk 6). Width/height read from the negotiated caps;
  pipeline always driven to NULL in a finally block.
- `FileSource`: Gst.Bin "source" = appsrc name=src (is-live=true format=time
  block=true do-timestamp=false max-bytes=8388608, extraction caps) !
  {h265|h264}parse name=parse ! nvv4l2decoder name=dec num-extra-surfaces=40,
  ghost pad "src" on dec. Feeder thread (name "feeder") pushes AUs cyclically:
  buf.pts = schedule_pts(...), dts=NONE, duration=au.duration; increments
  `_n_loops` itself at each wrap (single writer); loop=false → one pass then
  `end-of-stream` and exit; any non-OK push flow return (FLUSHING at
  teardown) exits the thread. `stop()` sets the flag, joins 1 s, and if the
  feeder is still blocked in a full-queue push sets the appsrc to NULL
  (flushing unblocks push-buffer) then joins — so stop() reliably returns
  even though it runs before pipeline NULL.
- `RtspSource`: rtspsrc ntp-sync=true add-reference-timestamp-meta=true
  drop-on-latency=true, pad-added → rtp{h265|h264}depay ! parse ! dec
  (num-extra-surfaces=40); is_live=True, never loops, no feeder; W×H via
  `model_cache.discover_size`.
- `create_source(config)`: select_variant → FileSource/RtspSource/registered
  factory. All Gst and deepstream_yolo imports are deferred into functions so
  tests.py can import the pure seams with only the package dir on sys.path.

Validation (all passing):

- `python3 -m py_compile ds_ros_pipeline/source.py` clean.
- Host, no Gst: pure-seam checks — loop_span derivation on a synthetic
  287-AU list; schedule_pts strictly monotonic and globally unique over
  4 wraps; variant selection incl. registered-scheme and ValueError paths;
  compute_loop_span([]) raises.
- In `deepstream-work:7.1` (docker run --rm --gpus all), against
  `streams/lorton-d4-rgb-nano.mp4`: extraction yields exactly the Sec 3.1/5
  numbers — 287 AUs, 119.5 MB, loop_span 9.5641 s, caps
  `video/x-h265, stream-format=byte-stream, alignment=au`, 2560×1440, pts
  468.7 → 9999.5 ms, duration 33_333_333 ns. max_preload_mb=1 guard fires.
  Bin audit: appsrc is-live/block true, do-timestamp false, max-bytes
  8388608, format=time, dec num-extra-surfaces=40, parse=h265parse, ghost
  pad "src" present, bin named "source".
- Live replay in-container (FileSource ! fakesink sync=false, free-running,
  probe on the ghost pad): loop=true over 2 wraps → 589 buffers
  (271 + 287 + partial; loop-0 RASL deficit as per Sec 5), 0 pts
  regressions, exactly 1 SEGMENT event, n_loops=2, stop() joined cleanly
  mid-loop. loop=false → 271 buffers then bus EOS, n_loops=0. h264: a
  generated 90-frame h264 mp4 extracts (h264parse chosen from caps) and
  replays to EOS with 90 buffers, 0 regressions.

Notes for testers:

- `n_loops` increments at the moment the feeder *finishes pushing* a pass
  (compressed domain); with `pace` attached the decoder/display lag makes
  the /ds/status increment appear near the ~9.6 s wrap cadence as designed.
- The wrap-RASL visual-artifact caveat (Sec 11 risk 11b) is unchanged; the
  optional skip-RASL-on-wrap mitigation is not implemented (design leaves it
  contingent on bring-up observation).
- file:// URIs with a netloc-relative path (the config default
  `file://streams/lorton-d4-rgb-nano.mp4`) resolve against the repo root
  (`deepstream_yolo.paths.PROJECT_DIR`); absolute `file:///...` also works.

## Interface notes

- `create_source`'s docstring says it discovers W×H via
  `model_cache.discover_size`, but the pinned constructors take only
  `(uri, config)` so discovery has to happen inside the variants:
  FileSource takes W×H from the negotiated extraction caps (identical
  values, no second pipeline) and falls back to `discover_size` only if the
  caps lack size; RtspSource uses `discover_size` directly. No signature
  changed; flagging the doc/behavior nuance for reviewers.
  - Tester round 1: ACCEPTED. The pinned `(uri, config)` constructors leave
    no other placement; caps-first with discover_size fallback is strictly
    better (no second pipeline) and matches Sec 9's dependency listing.

## Round 1 (adversarial tester)

No previously-open bugs to re-verify (this is the first tester round).

Independent re-validation of the coder's claims, all reproduced in
`deepstream-work:7.1` with `--gpus all` against
`streams/lorton-d4-rgb-nano.mp4` (scratch probe, not the coder's harness):
extraction = 287 AUs / 119,485,531 bytes / loop_span 9,564,144,445 ns /
2560×1440 / `video/x-h265` byte-stream caps; `max_preload_mb=1` guard and
missing-file RuntimeError both fire; loop=true free-run over 2 wraps = 709
buffers, 0 pts regressions, exactly 1 SEGMENT event on the ghost pad,
n_loops=2, `stop()` returns in ~0 ms mid-loop with the feeder joined;
loop=false = 271 buffers (Sec 11 risk 11a cold-start RASL discard) then bus
EOS, n_loops=0; `stop()` with the pipeline held in PAUSED and the feeder
blocked in a full 8 MiB appsrc queue returns in ~1000 ms (join timeout +
NULL-appsrc unblock) with the feeder joined — no deadlock. Pure seams
(compute_loop_span on reordered decode-order pts, schedule_pts
monotonic/unique over 4 wraps, select_variant incl. registered scheme and
ValueError, file:// path resolution incl. %-escapes) all pass on the host.
`python3 -m py_compile` clean. Design conformance walked line-by-line for
Sec 3.1 file/rtsp element names, properties, orderings, ghost pad, and
num-extra-surfaces=40 (range 0–55 confirmed via gst-inspect): conforming.

### BUG S1: RtspSource crashes at construction on DS 7.1 — `add-reference-timestamp-meta` does not exist in GStreamer 1.20.3 [major]

`source.py:409`. The image ships GStreamer 1.20.3; the
`add-reference-timestamp-meta` property landed on rtspsrc/rtpjitterbuffer in
1.22. Verified: `gst-inspect-1.0 rtspsrc | grep -ic reference` → 0 (same for
rtpjitterbuffer) in `deepstream-work:7.1`, and a live construction of
`RtspSource("rtsp://camera.local/stream", PipelineConfig())` (discover_size
stubbed to simulate a reachable server) raises
`TypeError: object of type 'GstRTSPSrc' does not have property
'add-reference-timestamp-meta'` from line 409. Concrete failure: any
`source.uri=rtsp://...` deployment crashes at startup inside
`create_source` — the entire rtsp variant is dead on this image.

Design rebuttal (evidence above): Sec 3.1 line 175 mandates the property,
but it is unsettable on this image, so the design is provably wrong on this
one item — the coder implemented it faithfully and inherited the crash.
Sec 5 already defines the sanctioned degradation: reference-timestamp-meta
is "preferred", with the arrival-time registry as fallback, so a guarded set
preserves design intent. Suggested contained fix in `RtspSource.__init__`:

    if rtspsrc.find_property("add-reference-timestamp-meta") is not None:
        rtspsrc.set_property("add-reference-timestamp-meta", True)
    else:
        print("source: rtspsrc lacks add-reference-timestamp-meta "
              "(GStreamer < 1.22); timestamps fall back to the ingest "
              "registry", flush=True)

`ntp-sync` and `drop-on-latency` do exist in 1.20.3 (verified) and stay
unconditional. Found by gst-inspecting every property the module sets.

> Coder round 2: FIXED as suggested — `find_property` guard around the
> `add-reference-timestamp-meta` set, else a one-line fallback notice
> (RtspSource.__init__; class docstring updated to match). Verified
> in-container (`deepstream-work:ds-ros`, GStreamer 1.20.3, discover_size
> stubbed): construction no longer raises, `find_property` returns None
> confirming the fallback path ran, ntp-sync/drop-on-latency still set
> unconditionally, ghost pad present.

### BUG S2: `file://localhost/...` URIs resolve under the repo root instead of the filesystem root [minor]

`source.py:114-124`. `_file_uri_to_path` treats any netloc as a relative
path prefix, so the RFC 8089-equivalent-to-local form
`file://localhost/abs/y.mp4` resolves to `PROJECT_DIR/localhost/abs/y.mp4`
(demonstrated on host) and fails the `is_file()` check with a confusing
"Source file not found: .../localhost/abs/y.mp4". The deliberate
netloc-relative convention for the config default is fine; just special-case
`netloc == "localhost"` to mean root. Found probing URI edge forms.

> Coder round 2: FIXED as suggested — `_file_uri_to_path` now maps a
> `localhost` netloc (case-insensitive) to empty before joining, so
> `file://localhost/abs/y.mp4` -> `/abs/y.mp4`. Host-verified alongside the
> unchanged behaviors: `file:///abs`, netloc-relative repo-root default,
> `%20` unquoting.

### BUG S3: RtspSource pad-added callback can AttributeError on caps-less pads and races on dual video streams [minor]

`source.py:416-428`. (a) `caps.get_structure(0)` returns None when
`query_caps` yields EMPTY caps (current caps unset); `structure.get_name()`
then raises AttributeError inside the rtspsrc streaming-thread callback —
pygobject swallows it with a printed traceback and the pad is silently never
linked. (b) Two simultaneous video-stream pad-added callbacks can both pass
the `is_linked()` guard and collide on the fixed names "depay"/"parse"
(second `bin.add` returns False, unchecked). Both are edge paths on
multi-stream/odd servers; a None-check plus ignoring non-first video
streams before element creation covers both. Found by code inspection of
callback threading.

> Coder round 2: FIXED. (a) The callback now returns when
> `caps.get_size() == 0` before touching the structure — this also avoids
> the GStreamer-CRITICAL that `get_structure(0)` on EMPTY/ANY caps logs
> (observed while reproducing; a bare None-check would have kept it).
> (b) The `is_linked()` guard is replaced by an atomic
> `claim.acquire(blocking=False)` (a closure-scoped Lock acquired once and
> never released), so exactly one video stream builds the depay chain even
> under concurrent callbacks. Verified in-container: emitting `pad-added`
> with a caps-less dangling pad returns cleanly with no traceback/CRITICAL;
> two threads emitting simultaneous H264 pads yield exactly one linked pad
> and exactly one "depay"/"parse" child in the bin.

### BUG S4: multi-video-track mp4 aborts extraction with a misleading link error instead of using the first track [minor]

`source.py:207-232`. A second qtdemux video pad re-creates elements with the
fixed names "extract_parse"/"extract_caps"; `pipeline.add` fails on the
duplicate name (return unchecked), the subsequent parentless `link` fails,
and extraction raises "Failed to link extraction parse chain" even though
track-1 extraction was proceeding. A `if link done: return` guard (mirror of
the rtsp `is_linked()` check) would take the first video track and ignore
the rest. Rare asset shape; found walking the pad-added error paths.

> Coder round 2: FIXED. Extraction's `on_pad_added` sets a first-video-track
> `linked` Event (single qtdemux streaming thread, so a flag suffices) and
> ignores later video pads; it also gained the same `caps.get_size()` guard
> as S3a. Verified in-container against a generated two-video-track mp4
> (2x 60-frame nvv4l2h264enc tracks in one mp4mux): extraction succeeds
> with 60 AUs / 320x240 from the first track, no link error. Standard-clip
> regression re-run: 287 AUs, loop_span 9564144445, 2560x1440.

Everything else hunted came up clean: feeder single-writer `n_loops`
(plain-int reads from the status thread are GIL-safe), stop()/teardown
ordering incl. the blocked-in-PAUSED case (demonstrated), push-buffer
FLOW-return exits, buffer pts/dts/duration schedule (Sec 5 verbatim),
appsrc property set (verbatim incl. max-bytes=8388608), extraction caps
string, config-interval=-1, `appsink.eos` semantics (returns TRUE only on
EOS+empty queue once started, so no lost tail samples), no floating-ref
leaks (add_pad/add sink the refs), extraction pipeline always driven to
NULL, and Gst-free importability of the pure seams for tests.py.

## Coder round 2

Fixed all four Round-1 bugs (per-bug notes inline above): S1 guarded
`add-reference-timestamp-meta` set with the Sec 5-sanctioned
ingest-registry fallback notice (design provably wrong on 1.20.3, tester's
evidence adopted); S2 `localhost` netloc means filesystem root; S3
`caps.get_size()` guard + atomic one-shot Lock claim in the rtsp
pad-added callback; S4 first-video-track guard (+ same caps guard) in the
extraction pad-added. Only `source.py` changed; no signatures touched.

Validation: `python3 -m py_compile` clean. Host: S2 URI-resolution matrix
(localhost/LOCALHOST, absolute, netloc-relative repo default, %-escapes)
passes. In `deepstream-work:ds-ros` (`docker run --rm --gpus all`):
RtspSource constructs on GStreamer 1.20.3 with the fallback notice printed
and ntp-sync/drop-on-latency set (S1); caps-less `pad-added` returns
cleanly, no CRITICAL (S3a); 2 concurrent H264 pad-added emits -> exactly 1
linked pad, 1 depay, 1 parse (S3b); a generated two-video-track mp4
extracts 60 AUs from track 1 (S4). Regressions: standard clip still
287 AUs / loop_span 9564144445 / 2560x1440; FileSource loop=false on the
generated h264 clip replays 60 buffers to bus EOS, n_loops=0, clean
stop()/NULL.

## Round 2 (adversarial tester)

Verification of previously-open bugs — all four re-checked independently
with my own probes (host: `/tmp/probe_pure_r2.py`; container:
`/tmp/probe_container_r2.py` run in `deepstream-work:ds-ros`,
`docker run --rm --gpus all`, GStreamer 1.20.3 confirmed), not the coder's
harness:

- **BUG S1: FIXED (verified round 2).** `source.py:419-429`: the
  `find_property` guard is present; `add-reference-timestamp-meta` is absent
  on this image's rtspsrc (re-confirmed via `find_property` at runtime), and
  `RtspSource("rtsp://camera.local/stream", PipelineConfig())` (discover_size
  stubbed) now constructs without raising, prints the exact
  "falls back to the ingest registry" notice on the fallback path, still sets
  `ntp-sync=true` and `drop-on-latency=true` unconditionally, has the "src"
  ghost pad, and `dec num-extra-surfaces=40`.
- **BUG S2: FIXED (verified round 2).** `source.py:119`:
  `file://localhost/abs/y.mp4` and `file://LOCALHOST/abs/y.mp4` ->
  `/abs/y.mp4`; `file:///abs`, the netloc-relative repo-root default, and
  `%20` unquoting all unchanged (host matrix).
- **BUG S3: FIXED (verified round 2).** `source.py:439-441` caps guard:
  emitting `pad-added` with an ANY-caps and an EMPTY-caps dangling pad
  returns cleanly — no traceback, no CRITICAL, no children added.
  `source.py:436,452` one-shot claim Lock: two barrier-synchronized threads
  emitting simultaneous H264 pads yield exactly one linked pad, exactly one
  "depay" and one "parse" child; a third late pad is ignored.
- **BUG S4: FIXED (verified round 2).** `source.py:208,212-219`: a freshly
  generated two-video-track mp4 (2x 60-frame nvv4l2h264enc tracks) extracts
  60 AUs / 320x240 / h264 byte-stream-au caps from the first track with no
  link error. Standard-clip regression intact: 287 AUs, 119,485,531 bytes,
  loop_span 9,564,144,445 ns, 2560x1440, h265 byte-stream/au caps,
  extraction <0.2 s.

New hunting (fix-adjacent code paths plus a fresh pass over concurrency,
teardown, and guards): `max_preload_mb=1` and missing-file RuntimeErrors
still fire; FileSource `loop=false` on a 60-frame h264 clip delivers exactly
60 buffers then bus EOS with `n_loops=0` and a ~0 ms `stop()`; `loop=true`
free-run on the standard clip reaches `n_loops=2` with 0 pts regressions,
exactly 1 SEGMENT event on the ghost pad, and a ~0 ms mid-loop `stop()`
with the feeder joined. Pure seams re-verified on host (loop-span
in-order + reorder-robust + empty-raises; schedule_pts monotonic/unique
over 4 wraps; variant selection incl. case-insensitive schemes, registered
extras, ValueError; a `register_variant("file", ...)` cannot shadow the
builtin since builtins are checked first). Also walked: the never-released
claim Lock is safe (plain Lock, no owner semantics); `Gst.Buffer.new_wrapped`
copies the Python bytes per push so re-pushing the same AU across loops is
sound (consistent with the multi-wrap runs); `stop()` before `start()`
returns immediately (`is_alive()` false on an unstarted thread);
`appsink.eos` cannot false-positive at startup because the sink's
READY->PAUSED (which sets started) runs synchronously inside
`set_state(PLAYING)` before the pull loop begins; sibling consumption of
the seam (`live_pipeline.SourceBin` usage, `ds_node` create/start ordering)
matches the pinned interface.

Non-bug observation (no action requested): in extraction's `on_pad_added`
the first-video-track `linked` flag is set before the codec-support check
(`source.py:218-221`), so an mp4 whose *first* video track is an
unsupported codec fails with "Unsupported video codec" even if a later
track is h264/h265. The pre-fix code also failed on such files (at EOS
instead of immediately), first-track-wins is the documented S4 policy, and
the error is clear — deterministic early failure is acceptable behavior for
a corner-of-a-corner asset shape.

No new bugs. **Round-2 verdict: clean** — zero blocker, zero major, zero
minor bugs open.
