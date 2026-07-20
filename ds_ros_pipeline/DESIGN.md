# ds_ros_pipeline — Signal-Driven DeepStream + ROS2 Pipeline Design

Status: DESIGN (round 4 — revised after adversarial review). Author: architect
agent. Implements nothing; every file named here except this document is
to-be-created.

Round-4 changes in brief: the round-3 recorder **stop** sequence (hold a
blocking probe on `q_rec.sink` through the whole branch drain) was
**empirically falsified** by the round-3 review
(`outputs/stop_order_check.py`, run in the `deepstream-work:7.1` container):
holding the block through the drain parks the tee's single streaming thread on
the probed pad — upstream of every leaky queue — so the preview branch
received 1 buffer in a 2 s window instead of ~60, and a disk stall mid-drain
(non-leaky `q_disk`) would hold the trunk forever. The same test also showed
the block cannot simply be lifted early: post-EOS buffers then reach the
EOS'd queue and its `FLOW_EOS` return stops the source through the tee. Stop
is **redesigned to the canonical momentary-block detach** (§7): an
explicitly-specified `IDLE` probe on the tee request pad whose callback
unlinks the branch, injects branch-local EOS, and releases the tee pad before
returning `REMOVE` — trunk hold time is the microseconds of the callback body
— with the drain completing **asynchronously** under a bounded timeout and a
force-finalize path that leans on the container formats' crash tolerance if
the disk is wedged. §3.2's isolation argument and the test plan (tests 9/16)
now cover the stop path explicitly. Also in this round: the test-file frame
accounting is stated precisely (315 container samples, 287 delivered after
qtdemux edit-list clipping — §3); a loop-wrap RASL pixel-correctness caveat is
documented (§5, §11 risk 11); the executor thread budget is raised and its
worst case analyzed (§4); and `CDCL_ROS_SETUP` is pinned to the new
workspace mount path (§2).

Round-3 changes retained: compressed-domain access-unit replay for file-source
looping (validated in-container, §5), fully specified `loop=false` EOS
behavior, Reentrant `grp_capture`. Round-2 changes retained: `identity
sync=true` trunk pacing; decoder surface-pool accounting
(`num-extra-surfaces=40`); validated batch-pipeline pool sizing
(`nvvideoconvert output-buffers=16`, `appsrc max-bytes=256 MiB`).

---

## 1. Overview

A single Python process (`ds_node.py`) runs **two independent GStreamer
pipelines** inside one ROS2 Humble node in one Docker container: a **live
pipeline** that decodes one swappable video source (file sources are replayed
in the compressed domain by an in-memory access-unit feeder, which is also
what makes seamless looping trivial), paces it to real time when the source is
non-live (emulating the future RTSP source), stamps every frame with
NTP-synced wall time at ingest, and fans out through leaky tee branches to a
continuous low-res preview, an on-demand frame grabber (mosaic/VLM/enqueue/
snapshot), and a dynamically attached disk recorder; and a **batch pipeline**
(appsrc → nvstreammux → yolo12x nvinfer → valve → injury-CLIP nvinfer) that
runs detection — and, only when signalled, assessment — over frames the grabber
queued, entirely decoupled from the live path so a running batch can never
stall it. All control is via ROS2 services; all outputs carry the ingest
timestamp. The live pipeline contains **no inference elements at all**, which
is what makes it lightweight and stall-proof.

---

## 2. Architecture & container topology

### Decision: ROS2 Humble inside the DeepStream container (single process)

Chosen over the existing two-container TCP-bridge split. Justification:

| Criterion | Single container (chosen) | TCP-bridge split (existing) |
|---|---|---|
| `/vlm_raw` (raw `sensor_msgs/Image`, ~11 MB/frame at 2560×1440 rgb8) | Direct `publisher.publish()`; DDS handles it | frame_wire protocol is JPEG-oriented; raw frames would add a serialize+copy+socket hop and a protocol extension |
| Signal latency ("publish the *next* frame") | Service callback flips a flag read by a pad probe in the same process — deterministic next-frame semantics | Signal must cross TCP; "next frame" becomes "next frame after a socket round-trip", racy |
| Recording/snapshot control | Service directly attaches/detaches Gst branches | Would need a new control channel in the wire protocol |
| Feasibility | DS 7.1 image is Ubuntu 22.04 (jammy) — exactly the distro ROS2 Humble targets; `apt install ros-humble-ros-base` is supported. Python 3.10 on both sides, so `rclpy` and `pyds` 1.2.0 coexist in one interpreter | Already works, but only for the JPEG-metadata flow |
| Cost | +~800 MB image layer; DS image and ROS deps now rebuild-coupled | Two processes, two failure domains, double memory for frames in flight |

Mitigations for the costs: the new image is a **layer on top of the existing
`deepstream-work:7.1`** (`FROM deepstream-work:7.1`), so the existing image and
all existing compose services are untouched, and the ROS layer sits last (cache
friendly, consistent with the project's layer-ordering convention). The
existing `ros-humble-publisher` / TCP-bridge services keep working unchanged —
nothing in `docker-compose.yml` is modified.

### Docker changes (new files only, all in `ds_ros_pipeline/`)

- **`ds_ros_pipeline/Dockerfile`** — `FROM deepstream-work:7.1`, then as root:
  add the ROS2 apt repo (ros.key + jammy main), `apt-get install
  ros-humble-ros-base ros-humble-vision-msgs ros-humble-diagnostic-msgs`,
  drop back to `USER user`. No `cv_bridge` (messages are built by hand from
  numpy — one less dependency); OpenCV-headless and numpy are already in the
  base image. `GIO_USE_PROXY_RESOLVER=dummy` is inherited from the base image's
  `ENV`.
- **`ds_ros_pipeline/compose.yaml`** — standalone compose file (invoked as
  `docker compose -f ds_ros_pipeline/compose.yaml ...`), defining one service
  `ds-ros-pipeline` with **`container_name: ds-ros-pipeline`** (so test
  commands like `docker kill ds-ros-pipeline` are project-prefix-independent):
  `build.context: ..`, `dockerfile: ds_ros_pipeline/Dockerfile`,
  `image: deepstream-work:ds-ros`, `network_mode: host`, `ipc: host`, GPU
  reservation, the repo bind mount at
  `${WORKSPACE_DIR:-/workspace/deepstream-work}`, and the external colcon
  workspace `${CDCL_ROS_WS:-/home/user/ros2_ws}` mounted read-only **at the
  path it was built at** (same symlink-install caveat the existing
  `ros-humble-base` service documents), plus `ROS_DOMAIN_ID`. Because the
  workspace mount path differs from the existing `docker-compose.yml`
  convention (which mounts at `/ros2_ws` and defaults `CDCL_ROS_SETUP` to
  `/ros2_ws/install/setup.bash` — a path that does **not** exist in this
  container), `compose.yaml` pins the setup path to match its own mount:
  `CDCL_ROS_SETUP=${CDCL_ROS_WS:-/home/user/ros2_ws}/install/setup.bash`
  (default `/home/user/ros2_ws/install/setup.bash`). The two values can never
  drift apart because both derive from the same `CDCL_ROS_WS` variable.
  Command: `bash -lc ds_ros_pipeline/run.sh`.
- **`ds_ros_pipeline/run.sh`** — sources `/opt/ros/humble/setup.bash` and
  `"$CDCL_ROS_SETUP"` (for `cdcl_umd_msgs`; set by `compose.yaml` as above —
  run.sh fails fast with a clear error if the file is missing rather than
  silently starting without `cdcl_umd_msgs`), sets
  `PYTHONPATH=$PWD/src:$PYTHONPATH` (to import `deepstream_yolo.*` helpers),
  then `exec python3 ds_ros_pipeline/ds_node.py "$@"`.

Process/thread model inside the container:

```
main thread          : GLib.MainLoop (Gst bus watch for both pipelines)
thread "feeder"      : file-source AU replay (owns n_loops; §3.1/§5) — file
                       variant only
thread "ros"         : rclpy MultiThreadedExecutor (num_threads=8, §4)
thread "batch"       : batch worker (pushes appsrc buffers, waits for results)
thread "disk"        : snapshot/one-shot-publish worker (PNG/JPEG encode, file writes)
Gst streaming threads: pad probes only touch locks, dicts, flags, and numpy copies
```

`rclpy` publishers are thread-safe and are called directly from Gst streaming
threads (preview) and from the workers (everything else).

---

## 3. GStreamer pipeline graph

Source resolution W×H is discovered at startup (`GstPbutils`, reusing
`deepstream_yolo.model_cache.discover_size`); for the test file W×H =
2560×1440, 30 fps, HEVC. (Frame accounting, precisely: the container track
really has **315 samples** — `ffprobe` reports `nb_frames=315` with packet pts
spanning **−0.5005 s → 9.9678 s** under the file's edit-list shift, verified
on `streams/lorton-d4-rgb-nano.mp4`. `qtdemux` **clips the ~28 leading
samples whose shifted pts fall before the segment start** — the clip was cut
from a longer stream — and delivers **287 access units**, pts 468.7 ms →
9 999.5 ms, 33.33 ms/frame; counted straight out of `qtdemux` in the DS 7.1
container. Neither number is "wrong"; 315 is what `ffprobe` sees in the
container, 287 is what the pipeline sees, and all loop/rate math below uses
287.)

### 3.1 Live pipeline (`Gst.Pipeline "live"`) — no inference elements

```
┌─ source bin "source" (built by source.py factory — THE one swap point) ──────┐
│ file variant — compressed-domain AU replay (loop-capable, validated §5):     │
│                                                                              │
│   [startup, one-shot extraction pipeline, runs to EOS in <2 s, then NULL:]   │
│     filesrc ! qtdemux ! {h265parse|h264parse} config-interval=-1             │
│       ! video/x-h26{5|4},stream-format=byte-stream,alignment=au             │
│       ! appsink sync=false                                                   │
│     → aus = [(bytes, pts, duration)] held in RAM (119.5 MB for the test      │
│       clip), plus the negotiated caps string and                             │
│       loop_span = last_pts − first_pts + frame_duration  (= 9.5641 s here)   │
│                                                                              │
│   [live elements:]                                                           │
│   appsrc name=src is-live=true format=time block=true do-timestamp=false     │
│       max-bytes=8388608 caps=<extraction caps>                               │
│   ! h265parse name=parse ! nvv4l2decoder name=dec num-extra-surfaces=40      │
│                                                                              │
│   feeder thread: pushes aus cyclically in decode order;                      │
│       buf.pts = au.pts + n_loops*loop_span; buf.dts = NONE (decode order is  │
│       the push order); n_loops incremented BY THE FEEDER at each wrap —      │
│       single-writer, no race, and it is the /ds/status loop counter.        │
│       loop=false → one pass, then appsrc end-of-stream (EOS behavior: §5).   │
│       NO seeks, NO SEGMENT_DONE, NO event dropping — appsrc emits exactly    │
│       one SEGMENT for the life of the process (measured).                    │
│                                                                              │
│ rtsp:  rtspsrc name=src ntp-sync=true add-reference-timestamp-meta=true      │
│           drop-on-latency=true ! rtp{h265|h264}depay ! parse                 │
│           ! dec (num-extra-surfaces=40)   — never loops, no feeder           │
│ (a third, IPC-shared-surface variant is a registered-but-unimplemented       │
│  factory slot — see §11 risk 5: GStreamer 1.20 in this image has no unixfd   │
│  plugin and no nvunixfdsrc element, verified by gst-inspect)                 │
│                                                                              │
│ Ghost src pad caps: video/x-raw(memory:NVMM), NV12, WxH                      │
└──────────────────────────────────────────────────────────────────────────────┘
      │
      ▼
   identity name=pace  sync={true if not is_live else false}  (see "Pacing")
      │  ◄── PROBE ingest_stamp (pace src pad):
      │      registry[buffer.pts] = time.time_ns()   (§5)
      ▼
   tee name=t_ingest  (allow-not-linked=false; every branch below hangs off a
      │                requested src_%u pad, each behind its own leaky queue —
      │                the tee itself therefore never blocks)
      │
      ├──[Branch G: grabber — serves mosaic / vlm_raw / enqueue / snapshot]────
      │   queue name=q_grab        leaky=2(downstream) max-size-buffers=4
      │                            max-size-bytes=0 max-size-time=0
      │   ! nvstreammux name=mux_grab batch-size=1 width=W height=H
      │        batched-push-timeout=40000 attach-sys-ts=false
      │        live-source={is_live} sync-inputs=false
      │   ! nvvideoconvert name=conv_grab nvbuf-memory-type=3   (CUDA unified —
      │        same dGPU-mappable-surface trick as the proven motion branch)
      │   ! capsfilter name=caps_grab
      │        caps=video/x-raw(memory:NVMM),format=RGBA,width=W,height=H
      │   ! fakesink name=sink_grab sync=false async=false enable-last-sample=false
      │       ◄── PROBE grab (on caps_grab src pad): checks armed flags/counters;
      │           when idle: return OK immediately (near-zero cost — the RGBA
      │           convert runs on-GPU; pixels only migrate to CPU when the probe
      │           actually maps the surface via pyds.get_nvds_buf_surface)
      │
      ├──[Branch P: continuous low-res preview — NOT signal-gated]─────────────
      │   queue name=q_preview     leaky=2 max-size-buffers=1
      │   ! nvvideoconvert name=conv_preview
      │   ! capsfilter name=caps_preview
      │        caps=video/x-raw(memory:NVMM),format=I420,width=640,height=360
      │   ! nvjpegenc name=enc_preview quality=75
      │   ! appsink name=sink_preview emit-signals=true sync=false
      │        max-buffers=1 drop=true
      │       → new-sample handler publishes /ds/preview/compressed
      │
      └──[Branch R: recorder — ATTACHED/DETACHED DYNAMICALLY on signal (§7)]───
          (h265 variant)
          queue name=q_rec         leaky=2 max-size-buffers=30
                                   max-size-bytes=0 max-size-time=0
                                   flush-on-eos=false
              ◄── PROBE rec_sidecar (q_rec src pad): appends {pts, ntp_ns} to
                  the .jsonl sidecar — runs only on frames that actually survive
                  the leaky queue, so the sidecar matches the file exactly
          ! nvvideoconvert name=conv_rec
          ! capsfilter name=caps_rec caps=video/x-raw(memory:NVMM),format=NV12
          ! nvv4l2h265enc name=enc_rec bitrate=200000000 control-rate=1(CBR)
               iframeinterval=30 idrinterval=30
          ! h265parse name=parse_rec config-interval=-1
          ! mpegtsmux name=mux_rec
          ! queue name=q_disk      leaky=0 max-size-bytes=67108864(64MiB)
               max-size-buffers=0 max-size-time=0
          ! filesink name=sink_rec sync=false async=false location=<outputs>/rec_<UTC>.ts

          (raw variant swaps everything after q_rec for:)
          ! nvvideoconvert name=conv_rec ! capsfilter caps=video/x-raw,format=I420
          ! queue name=q_disk leaky=0 max-size-bytes=268435456(256MiB)
          ! filesink name=sink_rec location=<outputs>/rec_<UTC>_WxH_I420.yuv
```

**Why AU replay instead of seek-based looping.** Round 2 looped by segment
seeks plus a `loop_normalize` probe that dropped SEGMENT events and rewrote
pts. The adversarial review implemented that mechanism faithfully in this
container and it failed: pacing drifted on the second pass and collapsed to
0.4 fps on the third; DROPping a sticky SEGMENT event never marks it
delivered, so GStreamer re-pushed it before every subsequent buffer (hundreds
to thousands of re-pushes); the `n_loops` increment had no race-free trigger;
and reseek routing fanned out to dynamically attached sinks. AU replay
removes the entire mechanism class: the *decoder never experiences a
boundary*. The feeder simply keeps pushing compressed access units whose pts
it assigns itself; each loop's first AU is the file's first AU (an IRAP with
in-band VPS/SPS/PPS thanks to `config-interval=-1`), so decode continuity
across the wrap is just normal streaming. RAM cost: the compressed AU list
(119.5 MB for this clip) — see §11 risk 6 for the long-file story. Validation
data in §5.

**Pacing (why `identity sync=true` exists).** Every sink in this pipeline is
`sync=false`, and a file source is non-live, so without a pacing element the
graph free-runs: measured in the `deepstream-work:7.1` container, `filesrc !
qtdemux ! h265parse ! nvv4l2decoder ! fakesink sync=false` consumes the whole
287-frame / 9.56 s clip in under 2 s (~200+ fps). That would loop the clip
every ~2 s, publish preview at ~200 Hz, demand ~70 Hz continuous inference at
stride 3, and outrun NVENC at 200 Mbps — none of which resembles the RTSP
source this file stands in for. The fix is a single clock-synced element **on
the trunk, upstream of the tee and of every leaky queue** (a `sync=true` sink
inside any branch could not pace the source, because the leaky queue in front
of it converts backpressure into drops): `identity name=pace sync=true` blocks
the source's streaming thread until each buffer's running time is reached
against the pipeline clock. Because the feeder presents one continuous
segment with monotonic pts, running time is monotonic across loops and pacing
stays correct forever — measured: 30.0 fps in every loop across a 4-loop run
(§5). Backpressure from `pace` propagates to `appsrc` (`block=true`,
`max-bytes=8 MiB`) and simply blocks the feeder thread — the feeder needs no
timer of its own. For live sources (`is_live`, e.g. rtsp) `pace` is created
with `sync=false` and is pure passthrough — arrival paces the pipeline
naturally. The `ingest_stamp` probe sits on `pace`'s **src** pad, so stamps
are taken at paced delivery time and inter-stamp spacing matches the emulated
live rate.

**Decoder surface-pool accounting (why `num-extra-surfaces=40` exists).**
`nvstreammux`, `tee`, and `queue` are zero-copy: every buffer queued in
`q_rec`, `q_grab`, and `q_preview` **is a decoder output surface**, pinned
until the branch's first converting element copies it. `nvv4l2decoder`'s
default `num-extra-surfaces` is 0 (range 0–55, verified in the DS 7.1 image),
leaving only the driver-minimum capture pool (~DPB+display ≈ 8 surfaces for
1440p HEVC). If pinned references ever exceed the pool, the decoder cannot
dequeue a capture buffer and decode blocks — stalling the tee and **every**
branch, defeating the leaky queues entirely (they only protect at the queue
level, not the pool level). Worst-case pinned-surface budget:

| Holder | Max pinned decoder surfaces |
|---|---|
| `q_rec` (deepest queue, leaky cap) | 30 |
| `q_grab` | 4 |
| `q_preview` | 1 |
| in-flight in tee/mux/convert/encoder front-ends | ~4 |
| **Total** | **~39** |

So `num-extra-surfaces=40` (≤ the 55 max) guarantees the decoder always has
its base pool free for decoding even with every branch saturated. VRAM cost:
2560×1440 NV12 ≈ 5.53 MB/surface → 40 × 5.53 ≈ **221 MB**, budgeted in §11
risk 1. Design rule for the factory: any future source variant that does not
expose an equivalent pool-size knob must instead insert a copy
(`nvvideoconvert` into an owned pool with `output-buffers` ≥ the table above)
**before** the tee, so deep branch queues never pin the producer's pool.
(Note: the repo's `configure_record_queue` precedent in
`src/deepstream_yolo/pipeline.py:65-75` looks similar but hangs off a tee
*after* `nvdsosd`, where buffers come from downstream conversion pools — that
precedent does not transfer to a tee fed directly by the decoder, hence the
explicit accounting here.)

### 3.2 Backpressure isolation argument

- Every branch is behind its **own** `leaky=2` queue on its **own** requested
  tee pad. A tee only blocks when a branch's sink pad blocks; a leaky-
  downstream queue never blocks its sink pad — it drops its oldest buffer
  instead. Therefore no branch, however slow or wedged, can stall the tee, the
  source, or a sibling branch **at the queue level**.
- **Pool level** (the failure mode queues cannot see): queued NVMM buffers pin
  the decoder's capture pool. `num-extra-surfaces=40` sizes the pool to the
  worst-case pinned total (§3.1 table), so pool exhaustion is impossible by
  construction. Both levels are required for the isolation claim to hold.
- Sinks: every `fakesink`/`appsink`/`filesink` has `sync=false` (never paced by
  the clock) and appsinks have `drop=true max-buffers=1`. The only clock-synced
  element is `pace` on the trunk, whose entire job is to pace (§3.1).
- The recorder is the only branch that wants depth: `q_rec` holds 30 NVMM NV12
  frames (1 s at 30 fps) and leaks as a last resort — a stalled disk degrades
  the recording (logged, counted) but never the pipeline. The post-mux
  `q_disk` is byte-bounded and **non-leaky**, because dropping muxed TS packets
  would corrupt the stream; backpressure from a slow disk propagates up to
  `q_rec`, which converts it into whole-frame drops in the raw domain — and
  the decoder rides on `num-extra-surfaces` while `q_rec` sits full.
  **The claim must also hold through detach**: stop (§7) blocks the trunk only
  for the microseconds of an `IDLE`-probe callback that unlinks the branch and
  releases the tee pad; the branch then drains **after** it is disconnected
  from the tee, so a slow — or permanently wedged — drain cannot reach the
  trunk by construction. (The naive alternative, holding a blocking probe
  through the drain, was measured to freeze the trunk for the whole drain and
  is explicitly rejected in §7.)
- Batch inference lives in a **separate `Gst.Pipeline`** (§6) with its own
  streaming threads; the only coupling to the live pipeline is GPU time-slicing
  (NVDEC/NVENC/CUDA are separate units for decode/encode/infer) and the grab
  probe's bounded numpy copies.

### 3.3 Batch pipeline (`Gst.Pipeline "batch"`) — persistent, idle until fed

```
appsrc name=src_batch is-live=true format=time block=true do-timestamp=false
     max-bytes=268435456 (256 MiB ≈ 18 RGBA frames; the 200 KB default would
     block the worker after the first 14.7 MB frame)
     caps=video/x-raw,format=RGBA,width=W,height=H,framerate=0/1
! nvvideoconvert name=conv_batch output-buffers=16
! capsfilter name=caps_batch caps=video/x-raw(memory:NVMM),format=RGBA
! nvstreammux name=mux_batch batch-size=8 width=W height=H
     batched-push-timeout=100000 attach-sys-ts=false live-source=false
! nvinfer name=pgie_batch config-file-path=ds_ros_pipeline/generated/ds_ros_infer_batch_yolo12x_640_640x384_b8.txt
     ◄── PROBE det_collect (src pad): reads obj metas per frame_meta,
         keys results by frame_meta.buf_pts (== feeder-assigned live-pipeline
         pts, globally unique — §5)
! valve name=v_assess drop=true          (drop=false only for *_assess runs)
! nvinfer name=sgie_batch config-file-path=configs/generated/config_infer_secondary_injury_clip_vit_l14_336_b8.txt
     process-mode=2 output-tensor-meta=true      (existing config + engine, unchanged)
     ◄── PROBE assess_collect (src pad): reuses
         deepstream_yolo.assessment_runtime.parse_assessment_tensor_meta
! fakesink name=sink_batch sync=false async=false
```

**Pool sizing is load-bearing, not tuning.** `nvstreammux batch-size=8`
accumulates up to 8 input buffers inside the mux before pushing a batch, and
those inputs are `conv_batch`'s pool buffers. `nvvideoconvert`'s default
output pool is 4 buffers — empirically (reproduced in `deepstream-work:7.1`
with a batch-size-counting probe), pushing 32 buffers through
`videotestsrc ! nvvideoconvert ! nvstreammux(batch-size=8) ! fakesink` with
default pools yields one partial batch `[4]` and then a permanent wedge (the
mux pins all 4 pool buffers waiting for 8; upstream blocks forever), while the
same pipeline with `output-buffers=16` produces clean `[8, 8, 8, 8]`.
`output-buffers=16` = 8 pinned in the mux + 8 for the next batch in flight.
Without this property the design's own `run_detect` with >4 queued frames
would hit the collector timeout and leave the persistent batch pipeline wedged
for every subsequent run.

The batch worker serializes runs: push k buffers → wait until `det_collect`
(and `assess_collect` if valve open) has seen k frames or a 10 s timeout →
publish → flip valve for the next run if needed. Because runs never overlap,
valve toggling is race-free. Queues of k>8 run as successive ≤8-frame batches;
a trailing partial batch is flushed by `batched-push-timeout=100000` (100 ms).

---

## 4. Signal interface (placeholder but concrete)

Node: `ds_pipeline` (namespace configurable, default none). All signals are
**services** (commands want acks; the response carries the observable result,
which makes every test in §10 scriptable). Service QoS: rclpy default
(reliable). Executor: `MultiThreadedExecutor(num_threads=8)`.

Thread-budget analysis (why 8): the deepest realistic pile-up is one
`grp_batch` run (≤30 s) + one `grp_record` stop (≤5 s bounded wait) + four
concurrent `grp_capture` calls (each ≤2 s + PNG write) = **6 occupied
threads**; `num_threads=8` leaves ≥2 threads free so `grp_fast` flips and the
1 Hz `/ds/status` timer stay schedulable through that worst case. Beyond it
(>4 simultaneous captures) the "sub-ms" latency of `grp_fast` is
**best-effort**: a flip may wait for a thread, bounded by the 2 s capture
timeout (self-healing, never deadlocked — no callback ever waits on another
callback's group). Accepted tradeoff, noted here rather than hidden.

Callback groups — blocking services must not serialize unrelated signals (or
each other, where the underlying machinery is concurrent), so groups are
split by blocking behavior; all state they share is mutex-guarded internally,
so cross-callback concurrency is safe:

| Group | Type | Services | Why |
|---|---|---|---|
| `grp_fast` | MutuallyExclusive | `enqueue`, `clear`, both `continuous_*` toggles | Non-blocking flag/counter flips; sub-ms |
| `grp_capture` | **Reentrant** | `capture/mosaic`, `capture/vlm`, `snapshot`, `snapshot_raw` | Each blocks up to ~2 s waiting for a frame (plus 200–600 ms PNG write for snapshots). Reentrant so a `capture/vlm` issued during a snapshot's write is served immediately — the arm/probe machinery can serve all four from the same frame; per-signal state is independently mutex-guarded, and file writes happen on the disk worker |
| `grp_record` | MutuallyExclusive | `record/start`, `record/start_raw`, `record/stop` | `stop` waits (bounded, `record.stop_timeout` default 5 s) on the branch's **asynchronous** drain — the wait occupies this service thread only, never the pipeline (§7); start/stop must serialize with each other only |
| `grp_batch` | MutuallyExclusive | `run_detect`, `run_detect_assess` | A 30 s batch must not block anything else |

| Name | Type | Semantics |
|---|---|---|
| `/ds/capture/mosaic` | `std_srvs/srv/Trigger` | Arm one-shot; the **next** frame through the grab probe is JPEG-encoded (full-res, quality 90) and published **once** on `/mosaic_compressed`. Call blocks (≤2 s) until published; `message` = the stamp used. |
| `/ds/capture/vlm` | `std_srvs/srv/Trigger` | Same, but publishes raw `rgb8` on `/vlm_raw`. |
| `/ds/batch/enqueue` | `std_srvs/srv/Trigger` | Increment pending-enqueue counter; each subsequent grab-probe frame is copied (RGBA numpy + its ntp stamp) into the batch queue until the counter drains. Response `message` = resulting queue depth; `success=false` if the queue (cap: `batch.capacity`, default 16) is already full. |
| `/ds/batch/clear` | `std_srvs/srv/Trigger` | Empty the batch queue (the *pending* queue only — a snapshot already taken by a running batch is unaffected, see race handling). |
| `/ds/batch/run_detect` | `std_srvs/srv/Trigger` | Valve `drop=true`; atomically **swap** the queue for a fresh empty one, run the swapped snapshot through the batch pipeline, publish one `TargetBoxArray` per frame on `/ds/detections`. Blocks until published (≤30 s). `success=false` if queue empty or continuous mode active. |
| `/ds/batch/run_detect_assess` | `std_srvs/srv/Trigger` | Same, valve `drop=false`; additionally publish `CasualtyImageCompressed` per detected person on `/ds/assessments`. Assessment runs **only** here / in continuous-assess mode. |
| `/ds/mode/continuous_detect` | `std_srvs/srv/SetBool` | `data=true`: grab probe auto-enqueues every `continuous.stride`-th frame (default 3 ≈ 10 Hz at the paced 30 fps); batch worker auto-runs whenever ≥`continuous.run_size` (default 4) frames queued or the oldest is >200 ms. `data=false`: stop. Manual run services are rejected while on. |
| `/ds/mode/continuous_detect_assess` | `std_srvs/srv/SetBool` | Same with the valve open. Turning either mode on turns the other off. |
| `/ds/record/start` | `std_srvs/srv/Trigger` | Attach the H.265/MPEG-TS branch (§7). `message` = file path. `success=false` if already recording or source ended. Recording runs seamlessly across loop boundaries (§5, §7). |
| `/ds/record/start_raw` | `std_srvs/srv/Trigger` | Attach the raw I420 branch instead. |
| `/ds/record/stop` | `std_srvs/srv/Trigger` | Momentary-block detach + asynchronous branch drain (§7) — the trunk is never stalled; `message` = path, frames written, frames dropped, and `drained=true|false` (false = drain timed out and the file was force-finalized, still valid by container choice). Idempotent after source-EOS finalization (§5): returns `success=true` with the stats of the already-finalized file. |
| `/ds/snapshot` | `std_srvs/srv/Trigger` | Next grab-probe frame → full-res PNG in `snapshot.output_dir`. Blocks until the file is closed; `message` = path. |
| `/ds/snapshot_raw` | `std_srvs/srv/Trigger` | Same, written as `.ppm` (raw RGB, header + pixels — trivially parseable, viewable) + `.json` sidecar with the stamp. |

Topics published:

| Topic | Type | QoS | Notes |
|---|---|---|---|
| `/mosaic_compressed` | `sensor_msgs/CompressedImage` | RELIABLE, KEEP_LAST 5, **TRANSIENT_LOCAL** | one-shot; durability latches the message so a subscriber joining after the publish still receives it (makes "echo then call" ordering non-fragile) |
| `/vlm_raw` | `sensor_msgs/Image` | RELIABLE, KEEP_LAST 1, **TRANSIENT_LOCAL** | ~11 MB/msg; depth 1 bounds latched memory to one frame |
| `/ds/preview/compressed` | `sensor_msgs/CompressedImage` | BEST_EFFORT, KEEP_LAST 1 (sensor-data profile) | continuous ~30 Hz, 640×360 JPEG q75 |
| `/ds/detections` | `cdcl_umd_msgs/TargetBoxArray` | RELIABLE, KEEP_LAST 10 | one per batched frame; `source_img` is a 640×368 JPEG of that frame (message-size hygiene, matches existing bridge) |
| `/ds/assessments` | `cdcl_umd_msgs/CasualtyImageCompressed` | RELIABLE, KEEP_LAST 10 | one per assessed person; `annotations` = 8 `clip_rgb_*` heads (same shape `ros_bridge.py` emits) |
| `/ds/status` | `diagnostic_msgs/DiagnosticArray` | RELIABLE, KEEP_LAST 1 | 1 Hz: state (`running`/`ended`), mode, queue depth, recording state, drop counters, loop count (= the feeder's `n_loops`) |

Race handling:

- **"Publish exactly the next frame once"**: arming is a mutex-guarded boolean
  + waiter list. The grab probe, under the same mutex, consumes the flag
  **before** copying, so exactly one frame is consumed per arming. Two calls
  that both arrive before the next frame **coalesce**: one message is
  published, both callers get `success=true` with the same stamp (documented
  behavior). A call arriving after consumption but before publish re-arms for
  the following frame. `enqueue` deliberately does *not* coalesce — it is a
  counter, so N calls between frames consume the next N distinct frames.
- **Enqueue during a run**: `run_detect*` (and the continuous worker) begin by
  atomically swapping the shared deque for a fresh empty one under the queue
  mutex; the run operates on the swapped-out snapshot. Frames enqueued while
  the run executes accumulate in the new deque and survive — there is no
  post-run clear that could destroy them. `clear` empties only the current
  (pending) deque.

ROS parameters (declared in `config.py`, all overridable via `--ros-args -p`):
`source.uri` (default `file://…/streams/lorton-d4-rgb-nano.mp4`), `source.loop`
(default true — AU-replay loop, §3.1/§5), `batch.capacity=16`,
`batch.engine_batch=8`, `continuous.stride=3`, `continuous.run_size=4`,
`preview.width=640`, `preview.height=360`, `preview.quality=75`,
`record.bitrate=200000000`, `record.output_dir=outputs/ds_ros`,
`record.stop_timeout=5.0` (seconds to wait for the asynchronous branch drain
before force-finalizing, §7),
`snapshot.output_dir=outputs/ds_ros`, `detections.image_width=640`,
`detections.image_height=368`, `frame_id=ds_camera`.

---

## 5. Timestamping

- **Capture**: the `ingest_stamp` pad probe on `pace`'s src pad records
  `registry[buffer.pts] = time.time_ns()` — the NTP-synced system clock (the
  container shares the host clock; keeping the host NTP-synced is an
  operational prerequisite, noted in README). Registry = bounded
  `OrderedDict` (2048 entries ≈ 68 s), single mutex.
- **Known bias, acknowledged**: this probe sits post-decoder, so for a live
  source without sender NTP the stamp includes decode latency and B-frame
  reorder delay (one to a few frame intervals, in presentation order rather
  than arrival order). For the file source this is moot (stamps are synthetic
  by construction — the probe placement at `pace` makes them *paced-delivery*
  time, the best emulation of a live camera). For RTSP the primary stamp is
  the **sender-side NTP** carried in-band (below), which has no such bias; the
  post-decode registry is only the fallback, and the bias is documented rather
  than hidden. If a future live source lacks sender NTP and the bias matters,
  the factory may move the registry probe to the depay/parse src pad (arrival
  order, keyed by the same pts, which decoders preserve) — a contained change
  inside `source.py`.
- **Key = feeder-assigned `buffer.pts`, globally unique**: the file-source
  feeder assigns `buf.pts = au.pts + n_loops × loop_span` in the compressed
  domain, and the decoder preserves pts through to presentation order, so pts
  is strictly monotonic for the life of the process — across loop iterations
  too (measured: 0 pts regressions over a 4-loop run, below). The tee shares
  the *same* buffer with every branch, and `nvvideoconvert`/`nvjpegenc`
  preserve pts, so any branch resolves its stamp by pts. In mux-bearing
  branches, `frame_meta.buf_pts` carries the same value. Uniqueness means the
  registry is **never flushed on loop** and batch result joining by pts (§6)
  cannot collide across iterations. (The first pts is the container's own
  first pts — 468.7 ms for the test clip, not 0 — which is irrelevant: the
  registry keys on actual values, and only monotonicity and uniqueness are
  load-bearing.)
- **Loop mechanism** (`source.loop=true`, file variant only) — **validated
  empirically in `deepstream-work:7.1`**, replacing the round-2 segment-seek
  design that the round-2 review falsified in the same container (drift on
  pass 2, 0.4 fps by pass 3, sticky-SEGMENT re-push storm):
  - *Mechanism*: §3.1's AU replay. One extraction pass at startup; then the
    feeder thread pushes compressed AUs cyclically into `appsrc` with
    self-assigned monotonic pts and `dts=NONE` (decode order = push order).
    There is no seek, no `SEGMENT_DONE`, no event manipulation anywhere;
    `n_loops` is incremented by the feeder itself at each wrap (single
    writer — the race the old design had around its increment trigger cannot
    exist).
  - *Validation harness*: extraction = `filesrc ! qtdemux ! h265parse
    config-interval=-1 ! video/x-h265,stream-format=byte-stream,alignment=au
    ! appsink` (yields 287 AUs, 119.5 MB, `loop_span = 9.5641 s`); playback =
    `appsrc (is-live=true format=time block=true max-bytes=8388608
    do-timestamp=false, extraction caps) ! h265parse ! nvv4l2decoder !
    identity sync=true ! fakesink sync=false`, feeder pushing 4 full loops,
    counting buffers/pts/SEGMENT events on the identity src pad.
  - *Measured results*: wall time 38.7 s vs 38.3 s expected (4 × 9.564 s);
    per-loop frame counts 271 / **287 / 287 / 287** at **30.0 fps each**;
    **0 pts regressions**; **exactly 1 SEGMENT event** downstream of the
    decoder for the entire run. The steady state is gapless and exact — no
    per-boundary frame shave (the reseek round-trip that caused it no longer
    exists). The 16-frame deficit in loop 0 only is the decoder discarding
    the clip's leading (RASL) pictures at cold start — standard HEVC
    CRA-start behavior, once per process, never at loop wraps.
  - *Scope of the validation — pixel correctness at wraps is NOT proven*: the
    287/287 counts mean those ~16 RASL pictures **are** emitted at every wrap
    (a mid-stream CRA does not flush the DPB, so the decoder keeps their
    references and outputs them). But those references are the *previous
    loop's tail frames*, not the pre-clip frames the RASL pictures were
    encoded against (the clip was cut from a longer stream — its first ~28
    container samples carry negative edit-list pts, §3). Timing and count are
    measured gapless; the first ~0.5 s after each wrap may therefore show a
    brief visual artifact in preview/recordings. One deliberate bring-up look
    plus a README note (next to the cold-start RASL note) — §11 risk 11.
  - *Consequences*: no EOS event and no FLUSH ever travels downstream at a
    loop boundary, so an attached recorder branch is never driven through mux
    finalization or a flushing restart; `pace` paces correctly forever
    (running time never regresses); the recorder's TS timeline stays
    monotonic through any number of loops. `rtsp` sources never loop — the
    feeder simply doesn't exist in that variant.
- **EOS behavior (`source.loop=false`)** — defined, since this is also the
  design's fallback mode: the feeder pushes one pass, then calls
  `appsrc.end_of_stream()`. The EOS flows downstream through the decoder
  (flushing out the DPB tail), `pace`, and the tee into **every** attached
  branch: an active recorder branch is finalized *by that EOS* (mpegtsmux
  writes out; the `.ts` is complete and valid — same drain path as an orderly
  stop), and the bus posts EOS only after all sinks received it. On bus EOS,
  `ds_node.py` transitions to state `ended` (visible in `/ds/status`, which
  keeps publishing) and does **not** exit: it wakes any armed capture/snapshot
  waiters with `success=false, message="source ended"`; subsequent
  `capture/*`, `snapshot*`, `enqueue`, `record/start*` calls fail fast the
  same way (no 2 s timeout); `record/stop` becomes idempotent and returns the
  already-finalized file's stats; `run_detect`/`run_detect_assess` **still
  work** on frames already in the batch queue (the batch pipeline is separate
  and unaffected); `clear` and the continuous toggles remain callable (the
  modes simply see no new frames). The process runs until SIGINT. Exercised
  by §10 test 15.
- **RTSP upgrade path**: when the source is later swapped to rtsp, the probe
  prefers `frame_meta.ntp_timestamp`/reference-timestamp-meta via the existing
  `deepstream_yolo.assessment_runtime.frame_timestamp()` helper plus
  `deepstream_yolo.frame_wire.is_wall_clock_timestamp()` (the same pairing
  `src/ros_bridge.py:32` uses; note `is_wall_clock_timestamp` lives in
  `frame_wire`, not `assessment_runtime`), falling back to the arrival-time
  registry. One function, `timestamps.resolve(pts)`, hides this.
- **Propagation**:
  - ROS: every `header.stamp` (and `CasualtyImageCompressed.stamp` /
    `position.header.stamp`) = `sec/nanosec` split of the resolved ntp ns —
    same copy-one-stamp-everywhere discipline as `ros_bridge.py`.
  - Batch: each queued `BatchItem(frame_rgba, ntp_ns, pts)` carries its stamp
    *with it*, so results map back even if the registry has since rolled;
    appsrc re-stamps pushed buffers with the original (feeder-assigned) pts →
    `frame_meta.buf_pts` → collector joins results to `BatchItem` by pts,
    which is collision-free because feeder-assigned pts is globally unique.
  - Disk: filenames embed UTC ingest time of the first frame
    (`rec_20260720T153001.123Z.ts`, `snap_20260720T153004.500Z.png`); every
    recording gets a `.jsonl` sidecar with one `{"pts":…, "ntp_ns":…,
    "utc":"…"}` line per frame actually written; snapshots get a `.json`
    sidecar with the same fields.

---

## 6. Batch inference

- **Queueing**: the grab probe (Branch G) copies the mapped unified-memory
  RGBA surface (`pyds.get_nvds_buf_surface` → `np.array(..., copy=True)`,
  ~14.75 MB/frame, a few ms) into a `deque` capped at `batch.capacity=16`
  (≈236 MB host RAM). Memory format: system-memory RGBA numpy + stamp +
  feeder-assigned pts. Copies happen **only** for signalled/continuous-strided
  frames.
- **Feeding**: appsrc → `nvvideoconvert` (host→NVMM) → `nvstreammux
  batch-size=8` → nvinfer, with the pool sizing of §3.3
  (`output-buffers=16`, `max-bytes=256 MiB`) — without which the default
  4-buffer convert pool deadlocks the b8 mux, as empirically reproduced.
  Chosen over TensorRT-direct because it reuses the exact letterboxing
  (`maintain-aspect-ratio=1 symmetric-padding=1`), the custom parser
  `lib/libnvdsinfer_custom_impl_Yolo.so`, and the SGIE crop-and-assess
  machinery for free, and keeps one inference stack in the repo. Queues >8
  frames run as successive ≤8-frame batches (`batched-push-timeout=100000`
  flushes partial batches after 100 ms).
- **Engines**:
  - **yolo12x: a batch-8 engine is needed** (only
    `yolo12x_640_640x384.onnx_b1_gpu0_fp16.engine` exists). No re-export
    required: `infer_configs.py` writes
    `ds_ros_pipeline/generated/ds_ros_infer_batch_yolo12x_640_640x384_b8.txt`
    — identical to the existing b1 primary config (same ONNX
    `models/yolo12x_640_640x384.onnx`, same labels snapshot, same custom
    parser, `net-scale-factor`, cluster settings) except `batch-size=8` and
    `model-engine-file=models/yolo12x_640_640x384.onnx_b8_gpu0_fp16.engine`.
    nvinfer builds and caches that engine on first start (~1–3 min, persisted
    via the bind mount). `src/deepstream_yolo/model_cache.py` is *not*
    modified (its writer hardcodes b1); this is the one piece of config logic
    the new folder duplicates, deliberately and minimally. **File-location
    note** (one-folder rule): the generated config lives under
    `ds_ros_pipeline/generated/` (gitignored runtime output), honoring the
    all-new-files-in-one-folder requirement; the engine, however, is written
    by nvinfer next to its ONNX in `models/` — that is nvinfer's own cache
    behavior and matches the repo's existing engine convention, called out
    explicitly in the README as the single deliberate exception.
  - **injury CLIP: existing b8 config + engine reused verbatim**
    (`config_infer_secondary_injury_clip_vit_l14_336_b8.txt`).
- **detect vs detect-assess**: `valve name=v_assess drop=true` sits between
  pgie and sgie. Detection results are collected on the pgie src pad *before*
  the valve, so detect-only runs cost zero CLIP compute; `_assess` runs open
  the valve. The worker serializes runs, so the valve never toggles with
  buffers in flight.
- **Result mapping**: `det_collect` keys per-frame results by
  `frame_meta.buf_pts`; the worker joins to `BatchItem`s by feeder-assigned
  pts (globally unique, §5) and publishes with each item's own ntp stamp.
  Assessment parsing reuses `parse_assessment_tensor_meta` (8 heads, softmax)
  unchanged.
- **Run lifecycle & races**: at run start the worker atomically
  snapshot-and-swaps the deque (§4); enqueues during the run land in the fresh
  deque and are never destroyed. There is no post-run clear.
- **Continuous modes**: same machinery — the mode flag makes the grab probe
  auto-enqueue every `continuous.stride`-th frame and the worker auto-run;
  no second code path. Stride default 3 (~10 Hz at the paced 30 fps) because
  yolo12x on an RTX 3070 is ~30–40 ms/frame — full 30 fps continuous is
  possible but leaves no GPU headroom (tunable to 1).
- **Isolation**: separate pipeline + `block=true` appsrc pushed only from the
  worker thread; if the GPU is saturated the *worker* waits, the batch queue
  caps, `enqueue` starts returning `success=false` — the live pipeline never
  sees any of it.

## 7. Recording & snapshot

- **Start** (`/ds/record/start`): request a `t_ingest` src pad, create the
  Branch R elements (§3.1), add to the pipeline, `sync_state_with_parent()`,
  link. Elements start in the running pipeline; no state change of the live
  graph.
- **Stop — momentary-block detach, asynchronous drain.** The round-3
  sequence (hold a blocking probe on `q_rec.sink` for the whole drain) is
  **rejected on measurement**: implemented verbatim in the DS 7.1 container
  (`outputs/stop_order_check.py` — live tee, leaky preview branch, deep
  `q_rec` behind a 20 ms/frame encoder stand-in), holding the block through
  the drain parks the tee's single streaming thread on the probed pad,
  upstream of every leaky queue: the preview branch received **1 buffer in
  2 s instead of ~60**. Every stop would freeze pacing, preview, captures and
  the feeder for the drain duration (~0.3–1 s h265, ~1 s+ raw), and a disk
  stall mid-drain (non-leaky `q_disk`) would hold the trunk **forever** — the
  exact failure §3.2 forbids, in the one path built to survive disk stalls.
  The same test rules out the obvious patch of unblocking early: post-EOS
  buffers then reach the EOS'd queue and its `FLOW_EOS` return propagates
  through the tee and stops the source. The replacement is the canonical
  dynamic-unlink recipe, in which the trunk is held only for the microseconds
  of one probe callback:
  1. Install a probe of type **`Gst.PadProbeType.IDLE`** (mask stated
     deliberately — *not* `BLOCK_DOWNSTREAM` on `q_rec.sink`, where a
     serialized EOS sent from the service thread into the blocked pad could
     itself block the stop call) on the **tee request pad** feeding the
     branch (`t_ingest.src_%u`). IDLE fires immediately if the pad is not
     mid-push, else right after the in-flight push completes.
  2. **Inside the callback**, in order: `tee_pad.unlink(q_rec.sink)`;
     `q_rec.sink.send_event(Gst.Event.new_eos())` — the pad is unlinked and
     unblocked, and `q_rec` is leaky (leaky queues never block enqueue, they
     drop), so the serialized EOS enqueues behind the buffered frames and
     returns immediately even if `q_rec` is sitting full against a stalled
     disk; `t_ingest.release_request_pad(tee_pad)`; return
     `Gst.PadProbeReturn.REMOVE`. Trunk hold time = this callback body.
     Ordering kills the `FLOW_EOS` hazard: no buffer can reach the branch
     after its EOS because the branch is already unlinked and the tee pad no
     longer exists.
  3. **Drain runs asynchronously**, disconnected from the tee: the EOS
     flushes `q_rec` → encoder → `mpegtsmux` → `q_disk` → `filesink`; an EOS
     probe on `sink_rec.sink` (installed at attach time) sets a `drain_done`
     event. The `grp_record` service thread — and only that thread — waits on
     it up to `record.stop_timeout` (default 5 s), then sets the branch
     elements to NULL, removes them from the pipeline, and reports stats.
  4. **Timeout ⇒ force-finalize**: if the drain does not complete (disk
     wedged under non-leaky `q_disk`), the service thread sets the branch to
     NULL anyway and reports `success=true, drained=false`. This is safe
     *because of* the container choices below: every byte that reached disk
     remains a valid playable TS (or raw-I420) prefix — the drain timeout
     path is deliberately identical in effect to the crash the formats were
     chosen to survive. The trunk never waited in either outcome.

  This EOS is branch-local by construction; and because the AU-replay loop
  never generates pipeline EOS or flushes (§5), there is no loop/stop race —
  with `loop=true`, stop's is the only EOS the branch can ever see. (With
  `loop=false`, the one pipeline-wide EOS at end of media finalizes the
  branch through the identical drain machinery — §5.) The detach sequencing
  (probe → unlink/EOS/release → async wait → NULL-or-timeout) is implemented
  once in `disk.py` as a pure state machine driven by injected callables, so
  its ordering and timeout path are unit-testable without GStreamer (§10).
- **Recording across loop boundaries** (defined semantics): with
  `source.loop=true`, the recorder sees one continuous monotonic stream —
  measured gapless at 30.0 fps across wraps (§5); a recording spanning N loop
  boundaries is a single valid TS file whose content repeats — timestamps in
  the file and the `.jsonl` sidecar keep increasing. No special casing, no
  restriction on test durations.
- **Crash safety** (the reason for MPEG-TS): a `.ts` file is a sequence of
  fixed 188-byte packets with PAT/PMT repeated (~every 100 ms) and **no
  trailer, index, or moov atom**. `kill -9` at any moment loses at most the
  tail after the last flushed write; everything before it is playable.
  `h265parse config-interval=-1` repeats VPS/SPS/PPS at every IDR and
  `idrinterval=30` gives 1 s closed GOPs, so a truncated file loses ≤1 s.
  CBR 200 Mbps (`control-rate=1`) keeps the disk load flat.
- **Raw variant** (`/ds/record/start_raw`): same dynamic attach and the same
  momentary-block detach/async-drain (the drain is longer — up to ~1.5 s of
  buffered raw at 166 MB/s — which is exactly why it must not run under a
  trunk block), branch tail =
  `nvvideoconvert → video/x-raw,I420 → q_disk → filesink *.yuv` + the same
  `.jsonl` sidecar. A headerless I420 stream is the ultimate crash-tolerant
  container: any 5,529,600-byte-aligned prefix is valid; the sidecar gives
  frame count/stamps; README documents the ffplay incantation
  (`-f rawvideo -pixel_format yuv420p -video_size 2560x1440`).
- **Disk math** (valid because the pipeline is paced to 30 fps — §3.1):
  H.265 200 Mbps = **25 MB/s** (1.5 GB/min, 90 GB/h) — any SSD; fine. Raw
  I420 2560×1440×1.5 B×30 fps = **166 MB/s** — needs a real SSD (SATA
  ~500 MB/s OK, NVMe comfortable, HDD not viable); `q_disk` (256 MiB) rides
  out ~1.5 s stalls, beyond that `q_rec` drops whole frames and counts them.
  NVENC on the RTX 3070 encodes 1440p **at the paced 30 fps** HEVC in hardware
  with ample headroom; 200 Mbps at 1440p30 is ~1.8 bits/px — visually
  lossless territory. (Unpaced, decode outruns NVENC at this bitrate — one
  more reason `pace` exists.)
- **Snapshot**: grab-probe copy → "disk" worker thread → `cv2.imwrite` PNG
  (lossless, ~200–600 ms for 2560×1440 — off the streaming thread, so
  harmless) or `.ppm` for raw. Filename + sidecar carry the stamp (§5).

## 8. Low-res continuous stream

Branch P: 640×360 (preserves 16:9), `nvjpegenc quality=75`, paced source rate
(~30 Hz, ~40–80 KB/frame ≈ 1.5–2.5 MB/s), topic `/ds/preview/compressed`,
BEST_EFFORT/KEEP_LAST 1 — a viewer that lags just misses frames. Stamped from
the registry like everything else. Resolution/quality are ROS parameters.

---

## 9. File manifest

All new files live under `ds_ros_pipeline/`, including runtime-generated
nvinfer configs (`ds_ros_pipeline/generated/`, gitignored). The single
exception is engine files, which nvinfer itself caches next to the ONNX in
`models/` (existing repo convention; documented in README).

| File | Purpose | Depends on (internal → external) |
|---|---|---|
| `README.md` | Nodes, topics/services, pipelines, usage, test commands, NTP + disk prerequisites, engine-location exception, RASL notes (cold start + wrap) | — |
| `DESIGN.md` | This document | — |
| `Dockerfile` | `FROM deepstream-work:7.1` + ros-humble-ros-base layer | existing `docker/Dockerfile` image |
| `compose.yaml` | `ds-ros-pipeline` service (container_name, GPU, host net, repo + ros2_ws mounts, `CDCL_ROS_SETUP` pinned to the ws mount path — §2) | `Dockerfile` |
| `run.sh` | Source ROS env(s), set PYTHONPATH, exec `ds_node.py` | `ds_node.py` |
| `config.py` | Dataclass of all parameters + ROS parameter declaration/defaults | → `rclpy` |
| `source.py` | Swappable source factory: uri → Gst.Bin (file/rtsp + registered slot for a future IPC source) with one NVMM src pad, `is_live`, `num-extra-surfaces`; file variant = one-shot AU extraction + appsrc replay feeder thread (owns `n_loops`, loop/EOS behavior of §5); the single place a source swap touches | `config.py` → `deepstream_yolo.media`, `model_cache.discover_size` |
| `timestamps.py` | `TimestampRegistry` (pts→ntp ns, bounded) + `resolve()` preferring stream NTP meta | → `deepstream_yolo.assessment_runtime.frame_timestamp`, `deepstream_yolo.frame_wire.is_wall_clock_timestamp` |
| `live_pipeline.py` | Builds the live graph: pace identity, t_ingest, Branch G, Branch P; exact queue/sink/pool settings of §3.1 | `source.py`, `timestamps.py` |
| `frames.py` | Grab probe: armed flags/counters, coalescing rules, `BatchItem` deque with snapshot-and-swap, surface→numpy copy | `timestamps.py`, `config.py` → `pyds`, numpy |
| `batch_pipeline.py` | Batch graph (appsrc→mux→pgie→valve→sgie→fakesink) with §3.3 pool sizing, worker thread, det/assess collectors, pts joining | `infer_configs.py`, `frames.py` → `deepstream_yolo.assessment_runtime` |
| `infer_configs.py` | Writes the b8 yolo12x primary config into `ds_ros_pipeline/generated/`; asserts ONNX/labels/parser lib exist | → `deepstream_yolo.paths`, existing `models/`, `lib/` |
| `disk.py` | Dynamic record branch attach + momentary-block detach with async drain/force-finalize as a unit-testable state machine (§7; h265-TS + raw), `.jsonl`/`.json` sidecars, PNG/PPM snapshot writers, disk worker thread | `live_pipeline.py`, `timestamps.py`, `config.py` |
| `ros_io.py` | The rclpy node: all services (semantics of §4), publishers, QoS profiles, four callback groups (incl. Reentrant `grp_capture`), message builders (borrowing the field mapping from `src/ros_bridge.py`), `/ds/status` timer, `ended`-state fail-fast behavior (§5) | `config.py`, `frames.py`, `disk.py`, `batch_pipeline.py` → `cdcl_umd_msgs`, `sensor_msgs`, `std_srvs`, `diagnostic_msgs` |
| `ds_node.py` | Entrypoint: Gst.init, build both pipelines, wire probes↔node, bus handling (incl. EOS→`ended` transition), thread startup/shutdown | everything above |
| `tests.py` | GPU/ROS-free unit tests (§10) | `timestamps.py`, `frames.py` (logic classes), `source.py` (pts-schedule function), `infer_configs.py` |

Existing code is imported, never modified: `deepstream_yolo.assessment_runtime`
(tensor parsing, `frame_timestamp`), `deepstream_yolo.frame_wire`
(`is_wall_clock_timestamp`), `deepstream_yolo.media` /
`model_cache.discover_size`, `deepstream_yolo.paths`, the custom parser `.so`,
the injury b8 config/engine, and the field-mapping conventions of
`ros_bridge.py`.

---

## 10. Test plan (against `streams/lorton-d4-rgb-nano.mp4`)

Bring-up: `docker compose -f ds_ros_pipeline/compose.yaml up --build
ds-ros-pipeline` (first run builds the b8 yolo engine, 1–3 min — watch the
log). All `ros2` commands below run in a second shell:
`docker compose -f ds_ros_pipeline/compose.yaml exec ds-ros-pipeline bash -lc
'source /opt/ros/humble/setup.bash && source $CDCL_ROS_SETUP && <cmd>'`.
`source.loop=true` (default) makes the clip (287 frames, 9.564 s span) loop
seamlessly forever — paced, one wall-clock loop every ~9.6 s (visible as a
`loop count` increment in `/ds/status` at that cadence, which is itself the
pacing observable). Expect a one-time deficit of ~16 frames in the very first
loop only (leading-picture discard at cold start, §5) — it is not a drop bug.

| # | Requirement | Exercise | Observable that proves it |
|---|---|---|---|
| 1 | Pacing + continuous preview | `ros2 topic hz /ds/preview/compressed`; watch `/ds/status` loop count | 29–30 Hz steady (NOT ~200 Hz — proves `pace` works); loop count increments every ~9.6 s wall; `ros2 topic echo --no-arr … header` stamps ≈ wall clock now, spaced ~33 ms |
| 2 | Mosaic one-shot | `ros2 topic echo /mosaic_compressed &` then `ros2 service call /ds/capture/mosaic std_srvs/srv/Trigger` (order-insensitive thanks to TRANSIENT_LOCAL) | Exactly one message per call; response `message` stamp equals the message `header.stamp`; no further messages without a new call; a late-joining echo still receives the latched message |
| 3 | VLM one-shot raw | `ros2 service call /ds/capture/vlm std_srvs/srv/Trigger`; `ros2 topic echo --no-arr /vlm_raw` | One `Image`, `encoding: rgb8`, `width: 2560, height: 1440`, `step: 7680` |
| 4 | Coalescing race | Fire two `mosaic` calls in the same shell command backgrounded together | Both succeed with identical stamp in `message`; exactly one topic message |
| 5 | Enqueue/run-detect (incl. >4 frames — the pool-sizing regression test) | 6× `ros2 service call /ds/batch/enqueue …` (responses show depth 1…6) then `ros2 service call /ds/batch/run_detect …` | Exactly 6 `TargetBoxArray` on `/ds/detections` (one 6-frame batch through the b8 mux — proves `output-buffers=16` fixed the empirical wedge), six distinct ascending stamps; person boxes plausible in Foxglove; depth back to 0 (per `/ds/status`) |
| 6 | Detect-assess gating | `run_detect` then `ros2 topic echo /ds/assessments` (nothing), then enqueue + `run_detect_assess` | Assessments appear **only** after the assess call; each has 8 `clip_rgb_*` annotations |
| 7 | Continuous mode | `ros2 service call /ds/mode/continuous_detect std_srvs/srv/SetBool "{data: true}"` | `/ds/detections` at ~10 Hz (stride 3 of paced 30 fps); preview hz unchanged at ~30 (isolation, §3.2); SetBool false stops it |
| 8 | Recording + crash tolerance | `/ds/record/start`; after ~6 s `docker kill ds-ros-pipeline` (SIGKILL, no drain; name is fixed by `container_name`) | `ffprobe outputs/ds_ros/rec_*.ts` → playable HEVC 2560×1440; duration within 1 s (one GOP) of kill−start wall time; sidecar line count ≈ ffprobe frame count |
| 9 | Recording clean stop **across a loop boundary** | start; wait 15 s (guaranteed ≥1 loop wrap at 9.56 s/loop); `/ds/record/stop` **while `ros2 topic hz /ds/preview/compressed` runs in a third shell** | Response reports path + frames ≈ 450 (15 s × 30 fps; §5 measured the wrap gapless, so no boundary tolerance needed beyond leaky-queue drops, which the response counts separately) and `drained=true`; `ffprobe` duration ≈15 s with a monotonic timeline (no discontinuity error at the wrap); bitrate ≈200 Mbps (`ffprobe -show_format` bit_rate ≈ 2.0e8); **preview hz never dips through the stop** (the round-3 drain-under-block design measurably dropped it to ~0.5 Hz — this observable pins the fix) |
| 10 | Raw recording | `/ds/record/start_raw`, 3 s, stop | File size ≈ 3×30×5,529,600 B (≈498 MB — valid math because paced); `ffplay -f rawvideo -pixel_format yuv420p -video_size 2560x1440 rec_*.yuv` shows the clip |
| 11 | Snapshot | `/ds/snapshot` and `/ds/snapshot_raw` | `snap_*.png` opens, 2560×1440; `.ppm` + `.json` sidecar stamp matches response |
| 12 | Timestamp propagation | Compare stamp of a mosaic, a detection for the same signalled frame, and the snapshot sidecar taken in the same second | Same clock domain (unix now), monotonic per source frame — including across a loop wrap (feeder-assigned pts never repeats) |
| 13 | Backpressure immunity | While recording *and* running continuous_detect_assess *and* echoing `/vlm_raw`, watch `/ds/status` and preview hz; also `nvidia-smi` for the VRAM budget | Preview stays ~30 Hz; loop cadence stays ~9.6 s (decoder pool never starves — proves the num-extra-surfaces accounting); drop counters may rise on recorder/batch only |
| 14 | Signal-timing independence | While a `snapshot` call is in flight (blocks ~0.5 s), fire `record/start`, `enqueue`, **and a `capture/vlm`** | All three return without waiting for the snapshot (separate groups for record/enqueue; Reentrant `grp_capture` for the concurrent capture, §4) |
| 15 | Non-looping EOS behavior (fallback mode, §5) | Relaunch with `-p source.loop:=false`; `/ds/record/start` at t≈2 s; wait past end of media (~10 s); then `/ds/capture/mosaic` and `/ds/record/stop` | `/ds/status` state flips to `ended` (still publishing at 1 Hz); the recording was finalized by the source EOS — `ffprobe` shows a valid ~8 s file; `capture/mosaic` returns `success=false, "source ended"` immediately (no 2 s hang); `record/stop` returns `success=true` with the finalized stats; process still alive |
| 16 | Stop under raw-drain load never stalls the trunk (§7) | `/ds/record/start_raw`; wait 3 s; `/ds/record/stop` while `ros2 topic hz /ds/preview/compressed` runs — the raw drain (up to ~1.5 s of buffered I420 at 166 MB/s) is the longest drain the design can produce | Preview hz shows no gap >~100 ms across the stop instant (the drain runs detached from the tee); loop cadence in `/ds/status` unperturbed; stop response arrives with `drained=true` ≤ `record.stop_timeout`. (The timeout/force-finalize path itself — a wedged disk — is exercised at the unit seam below, not with real hardware) |

**GPU/ROS-free seams** (run anywhere: `python3 -m pytest
ds_ros_pipeline/tests.py`): `TimestampRegistry` bounding and resolve-fallback
order; the feeder's pts-schedule function (pure: `(au_index, n_loops) → pts`,
asserting global monotonicity/uniqueness across wraps and the
`loop_span = last_pts − first_pts + duration` derivation against synthetic AU
lists); arm/coalesce/counter and snapshot-and-swap logic of `frames.py` (pure
state machines, probes injected as plain callables); the `ended`-state
fail-fast transitions of §5 as a pure state machine; the **record-detach
state machine** of `disk.py` (§7) with unlink/EOS/release/NULL as injected
callables — asserting the exact callback ordering, that no drain wait ever
occurs before the unlink+release, and the timeout → force-finalize path (a
drain_done that never fires); `infer_configs.py` output vs a golden config;
`source.py` uri→variant selection (bin construction mocked). Everything
touching `pyds`, NVMM, or `rclpy` is behind those seams and only exercised
in-container.

---

## 11. Risks & open questions

1. **VRAM headroom (8 GB)**: yolo12x FP16 b8 engine + CLIP ViT-L/14 b8 engine
   + decode/convert surfaces **+ 221 MB for the 40 extra decoder surfaces
   (§3.1)** may approach the limit. Fallbacks: `batch.engine_batch=4` (config
   knob; nvinfer builds a b4 engine instead), and/or shrinking `q_rec` to 15
   with `num-extra-surfaces=25` (halves both budgets at the cost of disk-stall
   tolerance). Measure at bring-up (`nvidia-smi` during test 13).
2. **`nvv4l2h265enc` 200 Mbps cap**: NVENC property ranges vary by driver;
   verify `bitrate=200000000` is accepted and actually sustained (test 9). If
   the v4l2 encoder caps lower, `nvh265enc` (NVCODEC element in the same
   image) is the fallback.
3. **`pyds.get_nvds_buf_surface` on RGBA unified memory without OSD** is
   assumed to work post-`nvstreammux` exactly as the existing motion branch
   proves — but that branch is nvof-fed; verify the grab branch mapping at
   bring-up (test 2 is the canary).
4. **11 MB `/vlm_raw` messages over DDS**: default rmw (Fast DDS) handles
   large messages but may need `udp_max_size`/shared-memory tuning if
   subscribers are remote. Localhost + `network_mode: host` should be fine;
   flag for the day a remote consumer appears. TRANSIENT_LOCAL latching holds
   one such frame resident per publisher — accepted (bounded, depth 1).
5. **Future IPC source variant**: DS 7.1's GStreamer is 1.20; the `unixfd`
   plugin (and any `nvunixfdsrc`) does not exist in this image (verified by
   gst-inspect — GStreamer's unixfd shipped in 1.24). The factory therefore
   registers file:// and rtsp(s):// today plus an explicit extension slot;
   the concrete element for zero-copy IPC ingest (shm, unixfd via a GStreamer
   upgrade, or an NVIDIA-provided source) is deferred until that swap is
   scheduled. Whatever lands there must honor the §3.1 pool rule (expose a
   pool-depth knob or copy before the tee).
6. **AU-replay RAM footprint scales with clip size**: the in-memory AU list is
   119.5 MB for this 10 s clip; a 10 min file at similar bitrate would be
   ~7 GB. Acceptable for the intended short looping test clips; if long files
   ever need looping, the extraction can spool AUs to a temp file and the
   feeder read them back (mechanism unchanged — only the storage behind the
   list). A `source.max_preload_mb` guard (refuse + log, suggesting
   `loop=false`) keeps this from surprising anyone. The loop mechanism itself
   is validated (§5) and carries no residual open question.
7. **`identity sync=true` pacing precision**: identity syncs to running time
   against the pipeline clock; measured 30.0 fps per loop in the §5 harness
   (fakesink-terminated); re-verify the 29–30 Hz observable in test 1 with the
   full branch set attached before trusting downstream rate math.
8. **Clock discipline is an external prerequisite**: the design stamps from
   the system clock; if the host loses NTP sync every stamp is wrong. `/ds/
   status` should surface `chrony`/offset info if available (nice-to-have).
9. **Continuous-assess at stride 1**: CLIP on every person on every frame can
   exceed 33 ms/frame with several people; continuous mode degrades gracefully
   (queue cap + enqueue skip) but sustained stride-1 assess is not a design
   target.
10. **First-run engine build (~1–3 min)** delays readiness; consider a
    `run.sh --prebuild` that runs `infer_configs.py` + a one-batch warmup
    before advertising services.
11. **RASL pictures: cold-start discard, and unverified pixels at wraps**:
    (a) the decoder drops the clip's ~16 leading (RASL) pictures once at
    process start (§5) — the first loop is 271/287 frames; harmless standard
    CRA-start behavior, bring-up should not chase it as a drop bug.
    (b) At every subsequent **wrap** those RASL pictures *are* emitted
    (287/287 measured), but a mid-stream CRA does not flush the DPB, so they
    decode against the previous loop's tail frames rather than the pre-clip
    frames they were encoded against (§3/§5) — timing is provably gapless,
    but the first ~0.5 s after each wrap (every 9.56 s) may show brief visual
    corruption in preview and recordings. Test-asset-only cosmetic issue (the
    rtsp variant never loops); take one deliberate look at a wrap during
    bring-up (test 1 preview or a test 9 recording spans one). If it offends,
    the contained fix lives in `source.py`: the feeder skips the leading RASL
    AUs on wrap iterations and shortens the per-wrap `loop_span` by their
    16 × 33.3 ms so the pts schedule stays gapless and monotonic.
    Both (a) and (b) go in the README side by side.

---

```json
{"folder": "ds_ros_pipeline", "files": [
  {"path": "ds_ros_pipeline/README.md", "purpose": "Usage, topics/services, pipelines, prerequisites (NTP, disk), test commands, engine-location exception, RASL notes (cold-start discard + possible wrap artifact)", "depends_on": []},
  {"path": "ds_ros_pipeline/DESIGN.md", "purpose": "This design document", "depends_on": []},
  {"path": "ds_ros_pipeline/Dockerfile", "purpose": "ros-humble-ros-base layer on top of deepstream-work:7.1", "depends_on": ["docker/Dockerfile"]},
  {"path": "ds_ros_pipeline/compose.yaml", "purpose": "ds-ros-pipeline service: container_name, GPU, host network, repo + ros2_ws mounts, CDCL_ROS_SETUP pinned to the ws mount path", "depends_on": ["ds_ros_pipeline/Dockerfile"]},
  {"path": "ds_ros_pipeline/run.sh", "purpose": "Source ROS envs, set PYTHONPATH, exec ds_node.py", "depends_on": ["ds_ros_pipeline/ds_node.py"]},
  {"path": "ds_ros_pipeline/config.py", "purpose": "All parameters: dataclass + ROS parameter declaration and defaults", "depends_on": []},
  {"path": "ds_ros_pipeline/source.py", "purpose": "Swappable source factory (file/rtsp + future-IPC slot) -> Gst.Bin with one NVMM src pad, num-extra-surfaces; file variant = startup AU extraction + appsrc replay feeder (owns n_loops, validated loop mechanism, loop=false EOS); the single source-swap point", "depends_on": ["ds_ros_pipeline/config.py"]},
  {"path": "ds_ros_pipeline/timestamps.py", "purpose": "TimestampRegistry (feeder-assigned pts -> NTP-synced wall ns) and resolve() preferring stream NTP meta", "depends_on": []},
  {"path": "ds_ros_pipeline/live_pipeline.py", "purpose": "Live graph builder: pace identity, t_ingest tee, grabber branch, preview branch, exact queue/sink/pool settings", "depends_on": ["ds_ros_pipeline/source.py", "ds_ros_pipeline/timestamps.py"]},
  {"path": "ds_ros_pipeline/frames.py", "purpose": "Grab probe state machine: one-shot flags with coalescing, enqueue counter, BatchItem deque with snapshot-and-swap, surface->numpy copy", "depends_on": ["ds_ros_pipeline/timestamps.py", "ds_ros_pipeline/config.py"]},
  {"path": "ds_ros_pipeline/batch_pipeline.py", "purpose": "Batch pipeline (appsrc->mux->pgie->valve->sgie) with explicit pool sizing, serialized worker, detection/assessment collectors keyed by feeder-assigned pts", "depends_on": ["ds_ros_pipeline/infer_configs.py", "ds_ros_pipeline/frames.py"]},
  {"path": "ds_ros_pipeline/infer_configs.py", "purpose": "Writes the batch-N yolo12x nvinfer config into ds_ros_pipeline/generated/ reusing existing ONNX/labels/custom parser; injury b8 config reused as-is", "depends_on": ["ds_ros_pipeline/config.py"]},
  {"path": "ds_ros_pipeline/disk.py", "purpose": "Dynamic record branch (H.265 MPEG-TS + raw I420): attach, momentary-block IDLE-probe detach with async drain and timeout force-finalize (unit-testable state machine), jsonl sidecars, PNG/PPM snapshot writers, disk worker", "depends_on": ["ds_ros_pipeline/live_pipeline.py", "ds_ros_pipeline/timestamps.py", "ds_ros_pipeline/config.py"]},
  {"path": "ds_ros_pipeline/ros_io.py", "purpose": "rclpy node: services, publishers, QoS (incl. TRANSIENT_LOCAL one-shots), four callback groups (Reentrant grp_capture), ended-state fail-fast, message builders, /ds/status", "depends_on": ["ds_ros_pipeline/config.py", "ds_ros_pipeline/frames.py", "ds_ros_pipeline/disk.py", "ds_ros_pipeline/batch_pipeline.py"]},
  {"path": "ds_ros_pipeline/ds_node.py", "purpose": "Entrypoint: builds both pipelines, wires probes to the node, bus handling incl. EOS->ended transition, thread lifecycle", "depends_on": ["ds_ros_pipeline/live_pipeline.py", "ds_ros_pipeline/batch_pipeline.py", "ds_ros_pipeline/ros_io.py", "ds_ros_pipeline/disk.py"]},
  {"path": "ds_ros_pipeline/tests.py", "purpose": "GPU/ROS-free unit tests: registry, feeder pts-schedule (monotonic/unique across wraps), arm/coalesce/swap logic, ended-state machine, config writer golden test, source selection", "depends_on": ["ds_ros_pipeline/timestamps.py", "ds_ros_pipeline/frames.py", "ds_ros_pipeline/source.py", "ds_ros_pipeline/infer_configs.py"]}
]}
```
