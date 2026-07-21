## Coder round 1

Implemented `live_pipeline.py` in full per DESIGN.md Sec 3.1/3.2:

- `build_live_pipeline`: `Gst.Pipeline "live"` = source bin -> `identity name=pace sync={not is_live}` -> `tee name=t_ingest allow-not-linked=false`; Branch G (`q_grab` leaky=2/max-size-buffers=4/bytes=0/time=0 -> `mux_grab` batch-size=1 WxH batched-push-timeout=40000 attach-sys-ts=false live-source={is_live} sync-inputs=false, linked via requested `sink_0` -> `conv_grab` nvbuf-memory-type=3 -> `caps_grab` NVMM RGBA WxH -> `sink_grab` fakesink sync=false async=false enable-last-sample=false) and Branch P (`q_preview` leaky=2/max-size-buffers=1 -> `conv_preview` -> `caps_preview` NVMM I420 preview.w x preview.h -> `enc_preview` nvjpegenc quality=preview.quality -> `sink_preview` appsink emit-signals=true sync=false max-buffers=1 drop=true). Each branch is on its own requested `src_%u` tee pad. All element names verbatim from Sec 3.1. No state changes, no probes installed by the builder (ds_node owns both).
- `install_ingest_stamp_probe`: BUFFER probe on `pace.src`, body = `registry.stamp(buffer.pts)`, returns OK; returns the probe id.
- `connect_preview`: new-sample handler pulls the sample, resolves ntp via `timestamps.resolve(registry, buffer.pts)` (the Sec 5 funnel), skips unresolvable frames, maps/copies the JPEG bytes, calls `publish(jpeg, ntp_ns)`, returns FlowReturn.OK.
- Recorder-attach seam for disk.py: `LiveParts.t_ingest` plus module helpers `request_tee_pad(t_ingest)` / `release_tee_pad(t_ingest, pad)` (additive — no pinned signature changed).
- `q_preview` deliberately sets ONLY leaky=2 + max-size-buffers=1 (Sec 3.1 states no bytes/time overrides for it; the 1-buffer cap always binds first).

Validation (both in `deepstream-work:ds-ros` with `--gpus all`, repo mounted at /workspace/deepstream-work):
1. `python3 -m py_compile` clean.
2. Structural check (throwaway /tmp/lp_check.py): built the real graph around a real `FileSource` on `streams/lorton-d4-rgb-nano.mp4` (287 AUs extracted, 2560x1440); asserted every Sec 3.1 element name resolvable via `get_by_name`, every property value above read back exactly (incl. leaky nick "downstream", nvbuf-memory-type=3, mux timeout 40000, fakesink/appsink flags), all static pads linked, tee holds exactly 2 request pads, and a `request_tee_pad`/`release_tee_pad` round trip (src_2 appears/disappears). PASS: `LIVE_PIPELINE_CHECK_OK`.
3. 6 s PLAYING smoke (throwaway /tmp/lp_smoke.py): full graph + probe + preview handler, feeder running. Result: `SMOKE_OK frames=148 fps=30.4 jpeg_first=60789B registry=152` — paced ~30 fps (not ~200, proving `pace sync=true`), no bus errors, plausible ~60 KB 640x360 JPEGs, every preview frame resolved a wall-clock stamp from the registry.

## Interface notes

None — implemented within the pinned skeleton; only additive helpers (`request_tee_pad`/`release_tee_pad`, three module constants) were added.

## Round 1 (adversarial)

No previously-open bugs to verify (this is the first adversarial pass; the file above is the coder's own log).

Independent verification performed (throwaway scripts in `/home/user/deepstream-work/outputs/`, run in `deepstream-work:ds-ros` with `--gpus all`, repo at `/workspace/deepstream-work`):

1. `python3 -m py_compile` clean; module import with `gi` blocked succeeds (Gst import really is deferred into the builders — the sibling-import claim in the module docstring holds).
2. `outputs/adv_lp_struct.py` — independent line-by-line Sec 3.1 conformance: exact child-element inventory (source + pace + t_ingest + 5 Branch G + 5 Branch P elements, nothing else), every property read back and compared (pace.sync=True for the file source, tee allow-not-linked=False, q_grab leaky=downstream/4/0/0, mux_grab batch-size=1 W=2560 H=1440 batched-push-timeout=40000 attach-sys-ts=False live-source=False sync-inputs=False, conv_grab nvbuf-mem-cuda-unified, caps_grab NVMM RGBA 2560x1440, sink_grab sync/async/enable-last-sample all False, q_preview leaky=downstream/1, caps_preview NVMM I420 640x360, enc_preview quality=75, appsink emit-signals/sync=False/max-buffers=1/drop=True), full link topology (each branch queue's sink peered to a distinct `t_ingest.src_%u`, q_grab feeding the requested `mux_grab.sink_0`, both chains fully linked, source->pace->tee), LiveParts handles are identity-equal to the pipeline's children, request/release round trip, builder leaves state NULL and installs zero probes on `pace.src` and `caps_grab.src`. Result: `ADV_STRUCT_OK`.
3. `outputs/adv_lp_smoke.py` — 6 s PLAYING with `install_ingest_stamp_probe` + `connect_preview`: 148 publishes at 29.4 fps measured between first/last ingest stamps (paced, not free-running), all payloads start `ffd8` (JPEG) at ~60-64 KB, stamp thread differs from publish thread (trunk vs preview streaming thread, per Sec 2 model), publish trails its stamp by median 2.7 ms (stamp taken at paced delivery, before downstream push — min lag +1.9 ms, never negative), zero pts regressions at the appsink, zero bus errors. Phase 2: removed the ingest probe so every new frame is unresolvable — publish was never called again and the pipeline kept running with no bus error, proving the skip path returns FlowReturn.OK. `ADV_SMOKE_OK`.
4. `outputs/adv_lp_teepad.py` — Sec 7 attach-seam claim: requested a `t_ingest` pad while PLAYING and left it dangling unlinked for 2 s (allow-not-linked=False), then released it. Preview flow stayed at exactly 30 fps through both windows, no bus errors. `ADV_TEEPAD_OK`.
5. Cross-module seams: `disk.py` consumes `LiveParts.t_ingest` (it inlines `request_pad_simple` rather than calling the helpers — cosmetic, disk.py's choice); `ros_io.publish_preview(jpeg: bytes, ntp_ns: int)` matches `connect_preview`'s `publish` contract exactly.

New findings:

### BUG LP-1: unresolvable preview frames are dropped with zero diagnostics [minor]
`live_pipeline.py:260-261` — `on_new_sample` returns OK silently when `resolve()` yields None. `timestamps.resolve`'s contract says "Returns None only if both miss (caller logs and skips the frame)" — this caller skips but never logs. Demonstrated (adv_lp_smoke phase 2): with no ingest stamps the preview topic goes permanently dark with no trace anywhere, indistinguishable from a dead branch; a registry roll-out burst (>68 s stall between stamp and publish) would likewise be invisible. Obvious fix: a rate-limited warning (e.g. count misses and log every Nth, or first miss) inside the None branch. Not upgraded to major because for the file source stamps are structurally guaranteed (probe upstream of the branch on the same buffer flow) and the coder's and my runs both resolved 100% of frames.

No other findings. Design conformance is verbatim on every Sec 3.1/3.2 requirement I could enumerate; concurrency surface is minimal (no shared mutable state in the module; registry has its own lock; handlers verified on distinct streaming threads); no state changes or probes leak from the builder; error paths (miss, failed map, None sample) all return OK and keep the pipeline alive. Module is clean apart from LP-1.
