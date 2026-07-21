# readme.md — review notes for ds_ros_pipeline/README.md (+ repo-root README section)

## Coder round 1

Implemented `ds_ros_pipeline/README.md` (new file) per DESIGN.md Sec 4/10/11
and the module brief, and added one short "Signal-Driven ROS2 Pipeline"
section to the repo-root `README.md` pointing at `ds_ros_pipeline/`
(inserted between "ROS Publisher" and "RTSP Timing"; nothing else in that
file touched).

Coverage in ds_ros_pipeline/README.md:

- What it is + ASCII architecture sketch (live pipeline with pace/tee/three
  branches; batch pipeline with valve), rationale deferred to DESIGN.md by
  link rather than duplicated.
- Prerequisites: yolo12x setup artifacts + injury b8 config, CDCL_ROS_WS
  workspace (mount-at-build-path caveat, run.sh fail-fast), **NTP sync**,
  **disk-speed table** (25 MB/s h265 / 166 MB/s raw) — Sec 11 notes.
- Bring-up: `docker compose -f ds_ros_pipeline/compose.yaml up --build
  ds-ros-pipeline`, exec-shell incantation, first-run b8 engine build note
  (~1–3 min, cached in models/), parameter-override example, link to the
  Sec 10 test matrix.
- Full service table (all 13, one-line semantics incl. coalescing vs
  counter, stop's drained=true|false, ended-state fail-fast) and topic table
  with QoS — checked 1:1 against `ros_io.py` registrations.
- `/ds/status` observables: state/mode/queue_depth/recording/loop_count plus
  the four drop counters, with exact string values from `frames.py`/`disk.py`
  (`off|detect|detect_assess`, `idle|recording|finalized`); loop_count
  cadence called out as the pacing observable.
- Parameter table: all 17 entries of `config.PARAMETER_MAP` with defaults
  verbatim (includes `source.max_preload_mb`, which config.py adds per
  Sec 11 risk 6).
- Disk outputs: filename patterns (`rec_<UTC>.ts`, `rec_<UTC>_WxH_I420.yuv`,
  `snap_<UTC>.png/.ppm`), `.jsonl`/`.json` sidecar line format, crash
  tolerance, and the raw **ffplay incantation** from Sec 7.
- Engine-location exception (one-folder rule), cold-start + wrap RASL notes
  side by side (Sec 11 risk 11), loop=false EOS semantics (Sec 5), and a
  troubleshooting section mapping symptoms to the above.

Validation: markdown only, so no py_compile; instead ran a cross-check
script that asserts every `/ds/*` service and one-shot topic registered in
`ros_io.py`, every `config.PARAMETER_MAP` name, all four `frames.counters()`
keys, and the config defaults (bitrate, source uri, output dirs) appear in
the README — all pass ("missing: none"). File paths, filename patterns, and
utc-tag format checked against `disk.py`/`timestamps.py`; compose/run.sh
commands checked against `compose.yaml`/`run.sh` as written.

No interface notes: the implemented interfaces I documented match DESIGN.md;
nothing in the skeletons contradicted the spec from the README's viewpoint.
Testers: test-matrix pointers in the README refer to DESIGN.md Sec 10 rather
than restating expected numbers, except the headline observables (29–30 Hz
preview, ~9.6 s loop cadence, 271/287 first loop) which are stated for
bring-up sanity.

## Round 1

No previously-open bugs to verify (this file had only the coder's notes).

Method: full read of DESIGN.md (all sections, esp. 4/10/11) and every
implementation file the README documents (config.py, ros_io.py, frames.py,
disk.py, timestamps.py, source.py, live_pipeline.py, batch_pipeline.py,
infer_configs.py, ds_node.py, run.sh, compose.yaml, Dockerfile, tests.py),
plus the repo-root README insertion. Executed checks:

- Independent cross-check script (`/tmp/readme_probe.py`): all 17
  `PARAMETER_MAP` names + the checkable defaults, all 13 service names, all
  6 topics, all 4 `frames.counters()` keys, status keys/state/mode/recording
  strings, and the disk filename patterns (driven through the real
  `disk.record_path`/`snapshot_path`/`timestamps.utc_tag`) all appear in the
  README verbatim — "problems: none".
- Prerequisite artifacts verified on disk: b1 primary config, ONNX, labels
  snapshot, parser .so, injury b8 config + engine, test clip; traced
  `scripts/setup.sh --model` → `prepare_models.py` → `model_cache
  .ensure_model` → `write_infer_config` to confirm setup.sh really produces
  the named config. Engine filename claim matches the generated b8 config's
  `model-engine-file` (checked the actual generated file).
- The `docker compose run --rm ds-ros-pipeline …` override example was
  probed live against Compose v5.3.0 with a throwaway compose file carrying
  `container_name`: `run` containers get generated names, so the example
  cannot collide with the `up`'d fixed-name container. The `exec`
  incantation's `$CDCL_ROS_SETUP` expansion (in-container env from
  compose.yaml) checks out.
- Numbers audited: 25 MB/s / 166 MB/s / 1.5 GB/min disk math, ~1.5 s raw
  q_disk ride-out (256 MiB / 166 MB/s), ~236 MB batch queue (16 × 14.75 MB),
  ~11 MB /vlm_raw, 287/9.564 s, 271/287, ~16 RASL, ~120 MB preload, 1 s
  GOPs, b8 engine name — all consistent with DESIGN/implementation.

### BUG readme-r1-1: /ds/preview/compressed QoS documented as KEEP_LAST 1, implementation publishes depth 5 [minor]

README.md:134 (topics table). The row says "BEST_EFFORT, KEEP_LAST 1
(sensor data)". ros_io.py:164-165 uses `qos_profile_sensor_data`, which I
verified in the `deepstream-work:ds-ros` container is BEST_EFFORT /
KEEP_LAST **5** / VOLATILE. Same self-contradiction as the design's own
"KEEP_LAST 1 (sensor-data profile)" cell — already filed on the code side
as rosio-r1-1 [minor]. Failure scenario: an operator tuning a viewer from
the README expects only-the-newest-frame delivery semantics and gets up to
5 buffered (~170 ms stale) frames under congestion. Whichever way
rosio-r1-1 resolves (explicit depth-1 profile, or keeping the named
profile), this table cell must end up agreeing with the code; if ros_io
keeps `qos_profile_sensor_data`, change the cell to "KEEP_LAST 5 (sensor
data)". Found by QoS conformance walk + in-container profile dump.

### BUG readme-r1-2: first-run engine-build behavior misdescribed — everything stalls, not just detection runs [minor]

README.md:75-77 ("services are advertised but detection runs wait until
nvinfer finishes") and README.md:246-248 (troubleshooting: "Long pause
before the first detection run"). In ds_node.py:179-186 the batch pipeline
goes PLAYING **first**, and nvinfer builds the b8 engine synchronously
inside that `set_state` call — before the live pipeline starts, before
`source.start()`, and before the executor thread spins (ds_node.py:188-198).
So on the very first run there is no preview, no `/ds/status`, and no
service *responses* (services are DDS-advertised from `build()` but no
callback executes) for the whole ~1–3 min build. Failure scenario: a
first-time user follows the very next README instruction — `ros2 topic hz
/ds/preview/compressed` with "Expect ~29–30 Hz" — sees nothing for minutes,
and the README has told them only *detection* waits, so they chase a
bring-up failure that isn't one. Fix: reword both spots to say the node
comes up only after the one-time engine build completes (the
"ds_node: pipelines PLAYING, services up" log line marks readiness), and
optionally mention the implemented-but-undocumented `run.sh --prebuild`
(ds_node.py:323-358) as the way to pay the cost ahead of time. Found by
tracing the ds_node.run() startup ordering against the README's bring-up
narrative.

### Notes (not filed as bugs)

- README.md:139-141 "Every `header.stamp` is the frame's resolved ingest
  time" is exact for the five frame-carrying topics; `/ds/status` (in the
  same table) stamps its header from the node clock at publish time
  (ros_io.py:522) — same NTP-synced wall clock, but not a frame ingest
  stamp. Below the bug bar; a two-word qualifier ("frame topics") would
  remove the ambiguity.
- README.md:283-286 documents `python3 -m pytest ds_ros_pipeline/tests.py`
  per DESIGN Sec 10; tests.py is currently a skeleton whose 13 tests all
  raise NotImplementedError, so the command fails today. That is tests.py's
  deliverable, not a README defect (the documented command is the
  design-mandated one) — flagged for the orchestrator since no review file
  for tests.py exists yet.
- The repo-root README "Signal-Driven ROS2 Pipeline" section (lines
  438-452) is accurate, links resolve, and sits between "ROS Publisher" and
  "RTSP Timing" as the coder stated; nothing else in that file changed
  around it.

Verdict: no blocker or major findings. Two minors open (readme-r1-1,
readme-r1-2).

## Master round 1

- **BUG MR1-README-1 (minor)** — README "Development" section documents only
  `python3 -m pytest ds_ros_pipeline/tests.py` as the way to run the unit
  tests ("run anywhere"), but pytest is not installed on this host and
  tests.py runs fine with plain `python3 ds_ros_pipeline/tests.py`
  (unittest). Observed: the documented command fails with "No module named
  pytest" on the host; expected: README should mention the dependency-free
  invocation (both work in-container; verified 15/15 pass both ways in the
  container). Everything else in the README was spot-checked as written and
  worked: bring-up command, exec incantation, the `docker compose run --rm
  ... -p source.loop:=false` override (works, subject to MR1-INFRA-1), the
  ffplay rawvideo flags (verified via the equivalent ffmpeg decode), engine
  cache location, RASL notes, and the troubleshooting entries.

## Master round 2

- **BUG MR2-README-1 (minor, carry-over of MR1-README-1 — still open, no
  fix applied).** README.md line 285 ("Development") still documents only
  `python3 -m pytest ds_ros_pipeline/tests.py` as the "runs anywhere"
  invocation; pytest remains absent on this host (`No module named
  pytest`), while `python3 ds_ros_pipeline/tests.py` passes 15/15 here.
  Add the plain-unittest invocation (or note the pytest dependency).
- Spot-checks this round: bring-up command, exec incantation, the
  `docker compose run --rm ... -p source.loop:=false` override (used for
  test 15 — works, and startup is now reliable after the MR1-INFRA-1 fix),
  and the raw-video ffplay flags (verified via the equivalent ffmpeg
  decode: 103/103 frames, rc=0) all work as written.
