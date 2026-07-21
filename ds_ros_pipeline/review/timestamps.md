# timestamps.py

## Coder round 1

Implemented `ds_ros_pipeline/timestamps.py` per DESIGN.md Sec 5, within the pinned skeleton interfaces (no signature changes):

- `TimestampRegistry`: bounded `OrderedDict[int, int]` (default `REGISTRY_CAPACITY = 2048`), one `threading.Lock` taken by every method, nothing else held. `stamp(pts)` records `time.time_ns()`, evicts oldest-first past capacity, returns the stamp. `get()` / `__len__` are lock-guarded reads. Never flushed — relies on feeder pts uniqueness per Sec 5.
- `resolve(registry, pts, frame_meta, buffer)`: when `frame_meta`/`buffer` are given, tries `deepstream_yolo.assessment_runtime.frame_timestamp` + `deepstream_yolo.frame_wire.is_wall_clock_timestamp`; a wall-clock source ("ntp"/"ref") with a non-None value wins; otherwise falls back to `registry.get(pts)`; None only on double miss. Both helpers are imported lazily via `_load_stream_helpers()` with a module-level cache (`_stream_helpers`); import failure is caught broadly and cached as None, so the module imports and resolves (registry-only) with no pyds/gi present. **Tester seam**: set `timestamps._stream_helpers = (fake_frame_timestamp, fake_is_wall_clock)` to exercise the preference order without pyds; reset to `timestamps._HELPERS_UNSET` to restore lazy loading.
- `split_stamp`: `divmod(ntp_ns, 1_000_000_000)`.
- `utc_tag`: integer-arithmetic `YYYYMMDDTHHMMSS.mmmZ` via `time.gmtime` (e.g. `20250720T151001.123Z`), truncating to milliseconds — matches the Sec 5 filename examples.

Validation:
- `python3 -m py_compile ds_ros_pipeline/timestamps.py` — clean.
- Host inline checks (no pyds): stamp/get roundtrip; 3000 stamps -> len == 2048 with newest-2048 retention; small-capacity eviction; resolve registry fallback and double-miss None; injected fake helpers confirm NTP-wins / non-wall-clock-falls-back / helper-None-falls-back ordering; split_stamp and utc_tag exactness. All passed.
- In-container (`docker run --rm --gpus all deepstream-work:7.1`, `PYTHONPATH=/work/src`): `_load_stream_helpers()` returns the real pair from `deepstream_yolo.assessment_runtime` / `frame_wire`; a fake frame_meta with NTP-epoch `ntp_timestamp` resolves to the unix-ns conversion (ntp path wins); `ntp_timestamp=0` + `buf_pts` falls back to the registry; miss returns None. Also verified without `--gpus` that the pyds ImportError path degrades gracefully to registry-only in the real image.

Notes for testers: `frame_timestamp` is getattr-based, so `frame_meta=None` or `buffer=None` are both safe to pass. `NS_PER_SEC` is exported for reuse.

## Interface notes

None — skeleton signatures were implementable as pinned.

## Round 1 (adversarial tester)

No previously-open bugs to verify (this is the first review round; the file contained only the coder's entry).

**Verdict: clean — zero findings.** Genuinely; details below so the next round can re-verify cheaply rather than re-derive.

Design-conformance walk (Sec 5, requirement by requirement):

- Registry = bounded OrderedDict, 2048 entries, single mutex: matches (`REGISTRY_CAPACITY = 2048`, one `threading.Lock` in every method, `stamp()` records `time.time_ns()` keyed by pts, oldest-first eviction). Never flushed on loop — no flush path exists at all, per the pts-uniqueness argument.
- Resolve order (RTSP upgrade path): exactly the design pairing — `deepstream_yolo.assessment_runtime.frame_timestamp` + `deepstream_yolo.frame_wire.is_wall_clock_timestamp` (verified the loaded objects are identical to the real module attributes in-container), stream wall-clock NTP preferred, registry fallback, None only on double miss. The "buf_pts"/"pts" sources `frame_timestamp` can return are correctly rejected by `is_wall_clock_timestamp` and fall through to the registry — writing them into a ROS header would date messages to 1970 (the exact hazard `frame_wire`'s comment warns about).
- Propagation helpers: `split_stamp` = exact divmod; `utc_tag` matches the Sec 5 filename examples (`YYYYMMDDTHHMMSS.mmmZ`, millisecond truncation) — cross-checked against `datetime` for boundary values (.000, .999, .500, epoch 0).
- GPU/ROS-free claim: holds — module imports stdlib only; helper import failure is cached as None and resolve degrades to registry-only (exercised on host with no gi).

Executed evidence (probe scripts, both exit 0):

- `outputs/ts_probe_host.py` (host, 29 checks): stamp/get roundtrip; 3000 stamps -> len 2048 with exact eviction boundary (951 gone, 952 kept); custom small capacity; resolve registry fallback and double-miss None; helpers NOT consulted when frame_meta/buffer are both None (poisoned fake would have raised); fake-helper preference matrix (ntp wins, ref wins, non-wall-clock falls back, ("ntp", None) falls back without crashing); seam reset to `_HELPERS_UNSET` re-triggers lazy load; split_stamp/utc_tag exactness; 2 s concurrency hammer (1 stamper + 4 resolvers, capacity 256) — no exception, capacity invariant held, all threads joined; duplicate-pts stamping cannot grow the dict past capacity.
- `outputs/ts_probe_container.py` (`docker run --rm --gpus all deepstream-work:7.1`, real helpers): NTP-epoch `frame_meta.ntp_timestamp` resolves to the correct unix ns; `ntp_timestamp=0` + valid `buf_pts` falls back to the registry; a **real `Gst.Buffer` carrying `ReferenceTimestampMeta`** (the `rtspsrc add-reference-timestamp-meta=true` mechanism) resolves via the "ref" path to the NTP->unix converted value — I specifically probed the swallowed-exception hazard in `reference_timestamp`'s `getter(None)` and it works on this image's PyGObject (returns the meta, no TypeError), so the design's mux-less RTSP wall-clock path is live, not silently dead; pts-only buffer falls back to registry; double miss -> None.

Considered and rejected as findings: unlocked lazy-init of `_stream_helpers` (benign idempotent race under the GIL; deployment imports cannot transiently fail since run.sh sets PYTHONPATH before exec); `stamp()` reading the clock before taking the lock (more faithful to capture time, not less); duplicate-pts OrderedDict position semantics (design guarantees pts uniqueness, and the probe shows no capacity corruption even if violated).
