## Coder round 1

Implemented `infer_configs.py` fully within the pinned skeleton (no signature
changes, no interface notes).

- `render_batch_yolo_config(N, repo_root)` (pure): parses the existing b1
  template `configs/generated/config_infer_primary_yolo12x_640_640x384.txt`
  (order-preserving, comments dropped — deterministic re-render, not a copy)
  and rewrites exactly two `[property]` keys: `batch-size=N` and
  `model-engine-file={onnx-file value}_b{N}_gpu0_fp16.engine` (engine naming
  duplicates `model_cache.artifacts_for_onnx`, generalized to N, per Sec 6).
  All other values — onnx, labels snapshot, parser lib, net-scale-factor,
  letterboxing, cluster settings, class-attrs — pass through verbatim, so the
  output inherits the template's absolute in-container paths
  (`/workspace/deepstream-work/...`), correct for the compose mount.
- `write_batch_yolo_config(8)` renders, then asserts `onnx-file`,
  `labelfile-path`, `custom-lib-path` exist; since the template embeds the
  container mount path, existence checks re-anchor each value at `repo_root`
  (split at the `models`/`lib`/`configs` component) so the check also works on
  the host checkout. FileNotFoundError lists every missing artifact with the
  value, the path actually checked, and the setup script that produces it
  (`yolo_export.sh` / `yolo_parser.sh`). Then creates
  `ds_ros_pipeline/generated/`, drops its `.gitignore` (`*` + `!.gitignore`)
  if absent, writes `ds_ros_infer_batch_yolo12x_640_640x384_b8.txt`.
  Default `repo_root` prefers `deepstream_yolo.paths.PROJECT_DIR`, falling
  back to `parents[1]` of the package when src/ is not on sys.path (tests.py
  only inserts the package dir; both resolve identically).
- `sgie_config_path()` returns the absolute path of the reused
  `config_infer_secondary_injury_clip_vit_l14_336_b8.txt`, with an
  `injury_model.sh` hint on FileNotFoundError.
- Repo file added: `ds_ros_pipeline/generated/.gitignore` (also self-healed at
  runtime by the writer).

Validation (host, no docker needed — pure file I/O): `python3 -m py_compile`
clean; a scripted run asserted (1) rendered b8 text differs from the b1
template in exactly the two designed lines, (2) render is deterministic,
(3) `write_batch_yolo_config()` writes the expected file + `.gitignore` and
`git check-ignore` confirms the txt is ignored, (4) parameterized N=4 yields
`batch-size=4` / `_b4_` engine, (5) a temp repo missing the parser `.so`
raises FileNotFoundError naming only `custom-lib-path` with the
`yolo_parser.sh` hint, (6) missing template raises with its own hint,
(7) `sgie_config_path` happy + missing paths, (8) default repo-root
resolution with and without `deepstream_yolo` importable. The generated b8
config is left in place at
`ds_ros_pipeline/generated/ds_ros_infer_batch_yolo12x_640_640x384_b8.txt`
for inspection (content is identical wherever it is rendered, since the
template embeds absolute container paths).

For testers: the golden test `test_infer_config_golden` should compare
`render_batch_yolo_config(8, repo_root)` against the b1 template with the two
substituted lines; note comments/blank-line normalization means the golden
should be the parsed-and-re-rendered form (the current b1 template has no
comments, so it is byte-identical minus the two lines).

## Round 1 (adversarial tester)

No previously-open bugs to verify (this is the first tester round; the section
above is the coder's implementation note, no rebuttals pending).

Verification performed (probe script `/tmp/probe_infer_configs.py`, 31 checks,
all passing; `python3 -m py_compile` clean; module is pure stdlib file I/O so
no in-container run needed):

- Design conformance, Sec 6 engines bullet, walked requirement by requirement:
  output path `ds_ros_pipeline/generated/ds_ros_infer_batch_yolo12x_640_640x384_b8.txt`
  (`batch_yolo_config_path(8)`) — conforms; output differs from the b1 template
  `configs/generated/config_infer_primary_yolo12x_640_640x384.txt` in **exactly
  two lines** (`batch-size=8`, `model-engine-file=…onnx_b8_gpu0_fp16.engine`),
  demonstrated by line-diff; everything else (onnx, labels snapshot, custom
  parser, net-scale-factor, letterboxing, cluster settings, class-attrs) passes
  through verbatim — conforms; engine filename matches nvinfer's own cache
  naming for `gpu-id=0`/`network-mode=2` (`<onnx>_b<N>_gpu0_fp16.engine`,
  generalizing `model_cache.artifacts_for_onnx`), so the first-start build is
  cached and re-deserialized — conforms; `model_cache.py` untouched — conforms;
  SGIE relpath `configs/generated/config_infer_secondary_injury_clip_vit_l14_336_b8.txt`
  matches Sec 3.3 — conforms. Sec 9: asserts onnx/labels/parser-lib existence
  with actionable setup-script hints — conforms (all three named in one error,
  verified). `generated/` is gitignored via self-healed `.gitignore` — conforms.
- Robustness actually executed: deterministic re-render; N=4 parameterization
  (risk-1 fallback) yields `batch-size=4`/`_b4_` engine; `engine_batch=0`
  rejected; a *future* template containing `#`/`;` comments (the current
  `deepstream_yolo.configs.write_infer_config` emits comment lines, though the
  on-disk template predates them) parses cleanly and renders identically;
  malformed templates (key before section, junk line, missing `[property]`,
  missing `batch-size`/`model-engine-file`) all raise with the template origin
  in the message — the `_require` guard on the two override keys prevents the
  silent-drop failure where a template lacking `batch-size` would otherwise
  render a config with no batch-size at all; missing-template and missing-SGIE
  errors carry their hints; failed artifact check writes nothing (no partial
  `generated/`); pre-existing `.gitignore` content is preserved; on-disk
  committed `generated/…_b8.txt` is byte-identical to a fresh render; default
  repo-root resolution identical with and without `deepstream_yolo` importable
  (`paths.py` is pure — no gi import — so `except ImportError` is the right
  breadth); 3.10-compatible syntax throughout.
- Concurrency: single call from the main thread at startup per Sec 9/ds_node
  skeleton; no shared mutable state; nothing to lock. Clean.

### BUG IC-1: existence check validates a re-anchored path but writes the verbatim template value [minor]

`infer_configs.py:100-111` (`_reanchor`) + `infer_configs.py:164-175`
(`write_batch_yolo_config`). The artifact check resolves each template value
against `repo_root` (splitting at the `models`/`lib`/`configs` component), but
the value written into the generated config is the template's verbatim string.
Two scenarios where the check passes yet the written config is dead on
arrival at nvinfer:

1. **Relative template values** (demonstrated by probe 9): a template with
   `onnx-file=models/a.onnx` passes the check (re-anchored to
   `<repo_root>/models/a.onnx`, exists) and is written verbatim — but nvinfer
   resolves relative config paths against the *config file's own directory*,
   which is now `ds_ros_pipeline/generated/`, not `configs/generated/`, so
   every path breaks at pipeline start. Unreachable today
   (`write_infer_config` always embeds absolute `PROJECT_DIR`-anchored paths),
   hence minor, but the check's silence is exactly wrong when it matters.
2. **Stale absolute mount** : template generated under the default
   `/workspace/deepstream-work` mount, container later started with a
   different `WORKSPACE_DIR`: `_reanchor` falls through to the repo-dirs split,
   finds the file under the real root, check passes — while the config still
   says `/workspace/deepstream-work/...` and nvinfer fails with its own less
   actionable error, defeating this module's fail-fast purpose. (Inherited
   hazard — the existing pipeline's configs break identically on a mount
   change — so not a conformance deviation, but this module is the one
   advertising the assertion.)

Obvious fix, either flavor: after `_reanchor`, if `resolved != Path(value)`,
either (a) raise with a "template path does not resolve as written; regenerate
the b1 config in this container" message, or (b) write `str(resolved)` into
the generated config instead of the verbatim value (deviates from "identical
except two lines" only when the verbatim value is already broken). Found by
code inspection of the check/write asymmetry, confirmed by executing probe 9.

No other findings: zero blocker, zero major. The module conforms to Sec 6 and
Sec 9 and survived every error-path and malformed-input probe thrown at it.
(Observation outside this module's scope: `tests.py`'s
`test_infer_config_golden` is still a `NotImplementedError` stub — the golden
seam of Sec 10 is owed by the tests.py coder, not by infer_configs.)
