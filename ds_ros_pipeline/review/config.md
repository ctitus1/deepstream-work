# config.py

## Coder round 1

Implemented `declare_parameters(node)` per DESIGN.md Sec 4 and the skeleton
docstrings; the `PipelineConfig` dataclass fields/defaults and `PARAMETER_MAP`
were already pinned by the skeleton and are unchanged (all 17 Sec 4 parameters,
defaults verbatim, plus `source.max_preload_mb` from Sec 11 risk 6 which the
skeleton pins at 1024).

What `declare_parameters` does:
- `node is None` -> returns `PipelineConfig()` defaults (standalone/unit-test
  mode, no ROS touched).
- Real node -> for each `PARAMETER_MAP` entry: `has_parameter` guard (makes
  re-calls idempotent, no `ParameterAlreadyDeclaredException`), then
  `declare_parameter(name, default)`, then reads `get_parameter(name).value`.
  Only duck-typed Node methods are used — the module never imports rclpy, as
  the skeleton requires.
- `_coerce(value, default)` normalizes the read value to the field's type
  (bool/int/float/str keyed off the dataclass default; `None` -> default).
  rclpy already type-checks declared parameters, so this only smooths benign
  cases like an int override for the float `record.stop_timeout`.

One deliberate change inside the pinned class: `@dataclass` ->
`@dataclass(frozen=True)`. The skeleton's module and class docstrings both
state the dataclass is frozen / never mutated after startup; the decorator
just didn't say it. Construction signature and field access are unchanged, so
no consumer is affected unless it mutates config fields — which the contract
forbids. If any module needs to mutate config, flag it here and I'll revert.

Validation:
- `python3 -m py_compile config.py` clean.
- Standalone script (host python, no ROS): PARAMETER_MAP keys <-> dataclass
  fields bijection (17/17); every Sec 4 default asserted verbatim;
  frozen-ness (`FrozenInstanceError` on assignment); `declare_parameters(None)`
  == defaults; duck-typed FakeNode with overrides (`source.loop=False`,
  `record.stop_timeout=2` int->2.0 float coercion, `preview.width=1280`)
  resolves overrides + declares the remaining 14 with defaults. All passed.
- In-container (`docker run --rm ... deepstream-work:ros-humble`, real rclpy
  Humble Node with `--ros-args -p source.loop:=false -p
  record.stop_timeout:=2.5 -p preview.width:=1280 -p frame_id:=cam0`):
  overrides resolved, untouched params kept defaults, second
  `declare_parameters` call idempotent and equal. Passed
  ("rclpy node checks OK").

## Interface notes

- None filed. `DEFAULT_SOURCE_URI = "file://streams/lorton-d4-rgb-nano.mp4"`
  is skeleton-pinned; note for the source.py coder/testers that its path part
  under strict RFC-3986 parsing is `/lorton-d4-rgb-nano.mp4` with netloc
  `streams` — `select_variant`/file-variant URI handling should treat
  everything after `file://` as a repo-relative path (or we switch the default
  to an absolute `file:///workspace/...` form). Config-side it is just a
  string default; flagging here so it is decided in source.py, not silently.
  - Tester round 1: concur, and keeping this open for source.py. DESIGN.md
    Sec 4 writes the default as `file://…/streams/lorton-d4-rgb-nano.mp4`
    (ellipsis = an absolute prefix), so the design intended an absolute-path
    URI; source.py is still a skeleton, so the interpretation cannot be
    verified yet. Not counted as a config bug.

## Tester round 1

No previously-open bugs to verify (this is the first tester pass).

Coder's deliberate change (`@dataclass` -> `@dataclass(frozen=True)`):
**accepted**. The skeleton's module/class docstrings state the config is
frozen and never mutated after startup; grepped every module in the package
for attribute assignment on a config instance (`config\.[a-z_]+ *=` over
`ds_ros_pipeline/*.py`) — zero hits, so no consumer is broken. Frozen-ness
independently re-verified (`FrozenInstanceError` on assignment).

What I checked and executed (probe scripts under /tmp, throwaway):

- Sec 4 conformance, row by row: all 16 Sec 4 parameters present with
  defaults verbatim *and exact types* (e.g. `record.stop_timeout` is the
  float 5.0, not int 5); `PARAMETER_MAP` keys <-> dataclass fields is a
  bijection (17/17 incl. `source.max_preload_mb`, which is sanctioned by
  Sec 11 risk 6 and skeleton-pinned at 1024). No Sec 4 parameter missing, no
  extra undesigned parameter. Sec 8's "Resolution/quality are ROS parameters"
  maps to `preview.*` — present.
- `python3 -m py_compile config.py` clean.
- Host probe: `declare_parameters(None)` == defaults; duck-typed FakeNode
  with overrides + a `None` parameter value (falls back to default) + second
  call performs zero redeclares and returns an equal config; `_coerce` edge
  matrix (bool-before-int ordering is correct — `bool` is an `int` subclass,
  so the order is load-bearing). All passed.
- In-container (`docker run --rm ... deepstream-work:ros-humble`, real rclpy
  Humble node): dotted-name declares + CLI overrides
  (`source.loop:=false`, `record.stop_timeout:=2.5`, `preview.width:=1280`,
  `record.bitrate:=100000000`, `frame_id:=cam0`) all resolve, untouched
  params keep defaults, second `declare_parameters` call idempotent and
  equal. Passed.
- Concurrency: frozen instance, no locks needed, built once on the main
  thread before the executor spins — nothing to race.

## Round 1

### BUG CFG-1: `_coerce`'s int->float smoothing is unreachable with real rclpy; an integer CLI override for a double parameter crashes startup [minor]

config.py:87 / config.py:91-105.

Demonstrated in-container (deepstream-work:ros-humble, real rclpy Humble):

- `--ros-args -p record.stop_timeout:=2` raises
  `InvalidParameterTypeException: Trying to set parameter
  'record.stop_timeout' to '2' of type 'INTEGER', expecting type 'DOUBLE'`
  inside `node.declare_parameter` at config.py:86 — before `_coerce` ever
  runs. The docstring's claimed normalization of "int given for a float
  field" (config.py:94) can therefore only happen with duck-typed test
  nodes, never in production; a user who types the natural `-p
  record.stop_timeout:=2` gets a traceback at node construction instead of a
  working override. (Mirror case: `-p preview.width:=1280.0` raises the same
  way for INTEGER fields.)
- Why minor and not major: the failure is loud, immediate, and
  self-explanatory (standard ROS 2 static-typing behavior for every Humble
  node; the design's own docs write the default as `5.0`, steering users to
  the float literal). No wrong value is ever silently accepted.
- Related footgun in the same function, same severity: with a duck-typed
  node, `_coerce("false", True)` returns `True` (`bool` of a non-empty
  string). Unreachable with rclpy for the same reason; worth a guard only if
  tests ever feed string overrides.
- Obvious fix (coder's choice): wrap the `declare_parameter` call and
  re-raise with a one-line message naming the parameter and expected type
  (fail-fast kept, traceback tamed), and trim the `_coerce` docstring to
  claim only what is reachable (`None` -> default, bool/int/float/str
  normalization for duck-typed nodes). Alternatively just fix the docstring.

No blocker or major findings. Apart from CFG-1 the module conforms to Sec 4
exactly and survived every adversarial probe I could construct; it is a
frozen value object with no concurrency, resource, or error-path surface
beyond the one documented above.
