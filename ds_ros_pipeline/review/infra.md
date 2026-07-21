# infra review — Dockerfile / compose.yaml / run.sh

Tester round 1 (adversarial). No prior review file existed; there are no
previously-open bugs to re-verify. Governing spec: DESIGN.md Sec 2 (docker
changes, CDCL_ROS_SETUP pinning, container_name), Sec 9 manifest rows for the
three files, Sec 10 bring-up/test commands that depend on them.

## Round 1

### Conformance walk (Sec 2, requirement by requirement)

Dockerfile:
- `FROM deepstream-work:7.1` — present (line 17); base image verified locally
  (USER=user uid 1000, `GIO_USE_PROXY_RESOLVER=dummy` and
  `DEBIAN_FRONTEND=noninteractive` baked in ENV, WorkingDir
  `/workspace/deepstream-work`), so the "inherited from the base image's ENV"
  claim is true.
- root → add ROS2 apt repo (ros.key + jammy main) → install
  `ros-humble-ros-base ros-humble-vision-msgs ros-humble-diagnostic-msgs` →
  `USER user` — all present. The install is split across two RUN layers
  (ros-base alone, then the two msg packages); the design writes it as one
  apt-get line but the split is behavior-identical and follows the project's
  late-layer cache convention — accepted, not a finding.
- No cv_bridge installed — conforms.

compose.yaml (checked against `docker compose -f ds_ros_pipeline/compose.yaml
config` output): one service `ds-ros-pipeline`, `container_name:
ds-ros-pipeline` (test 8's `docker kill` target), `build.context: ..` +
`dockerfile: ds_ros_pipeline/Dockerfile` (resolves to repo root / correct
file), `image: deepstream-work:ds-ros`, `network_mode: host`, `ipc: host`,
nvidia device reservation, repo bind mount at
`${WORKSPACE_DIR:-/workspace/deepstream-work}` (rw — needed for outputs/ and
the engine cache), ros2_ws mounted read-only at its build path
`${CDCL_ROS_WS:-/home/user/ros2_ws}`, `ROS_DOMAIN_ID`, and
`CDCL_ROS_SETUP=${CDCL_ROS_WS:-/home/user/ros2_ws}/install/setup.bash` — both
sides derive from the same variable, so the mount and the setup path cannot
drift (resolved config shows `/home/user/ros2_ws/install/setup.bash` against a
`/home/user/ros2_ws:/home/user/ros2_ws:ro` mount). Command is exactly
`bash -lc ds_ros_pipeline/run.sh`. All conform.

run.sh: sources `/opt/ros/humble/setup.bash`, fail-fast on unset AND on
missing `$CDCL_ROS_SETUP` (design only demanded the missing-file case; the
unset case is a strict improvement), sources the overlay, prepends `$PWD/src`
to PYTHONPATH, `exec python3 ds_ros_pipeline/ds_node.py "$@"`. Uses
`set -eo pipefail` without `-u` — correctly avoiding the documented
ament/colcon unbound-variable pitfall that scripts/ros_service.sh has to
work around with `set +u/set -u` pairs. Conforms.

### Executed validation (evidence)

1. `docker compose -f ds_ros_pipeline/compose.yaml config` — parses; all
   interpolations resolve as designed (defaults and container_name verified).
2. Dockerfile RUN-1 body executed verbatim in `deepstream-work:7.1` as root:
   ros.key downloads as a binary GPG keyring, the signed-by jammy repo line
   passes `apt-get update` signature verification, and
   `apt-get install --dry-run` resolves all three packages from the ROS repo
   (ros-base 0.10.0, vision-msgs 4.1.1, diagnostic-msgs 4.9.1). The
   transaction includes ros-humble-sensor-msgs / std-msgs / std-srvs (the
   node's other message deps ride in with ros-base's common_interfaces).
3. Full `docker compose -f ds_ros_pipeline/compose.yaml build` — succeeds,
   produces `deepstream-work:ds-ros`.
4. In the built image (`--gpus all`): `whoami`=user; Python 3.10.12; one
   interpreter imports rclpy + pyds + Gst + sensor_msgs + std_srvs +
   diagnostic_msgs + vision_msgs + numpy + cv2 (the Sec 2 coexistence claim,
   demonstrated); `GIO_USE_PROXY_RESOLVER=dummy` present.
5. GStreamer stack regression check in the built image: `nvv4l2decoder` and
   `nvjpegenc` still inspectable, `Gst.init` OK (1.20.3) — the ROS layer did
   not disturb the DS stack.
6. run.sh behavioral matrix, run in-container with the repo mounted at the
   compose workdir:
   - CDCL_ROS_SETUP unset → clear refusal, rc=1;
   - CDCL_ROS_SETUP → missing file → clear refusal naming the path, rc=1;
   - compose-equivalent env with the REAL host colcon overlay
     (`/home/user/ros2_ws/install/setup.bash`, mounted ro at its build path)
     → both setups source cleanly under `set -eo pipefail`, PYTHONPATH comes
     out `$PWD/src:<overlay dist-packages>:<humble paths>` (repo first), args
     are forwarded verbatim to `exec python3 ds_ros_pipeline/ds_node.py`, and
     `import cdcl_umd_msgs.msg` succeeds through the overlay
     (TargetBox/TargetBoxArray present).
   (Note: an earlier run of the same matrix in `deepstream-work:ros-humble`
   showed `src` twice in PYTHONPATH — that duplicate comes from that test
   image's own baked `ENV PYTHONPATH`, absent from `deepstream-work:7.1`; not
   a run.sh defect.)
7. No CRLF line endings in any of the three files; run.sh is executable.

### BUG INFRA-1: keyring curl lacks `--fail`, so an HTTP error page becomes the keyring [minor]

ds_ros_pipeline/Dockerfile:23. `curl -sSL <ros.key> -o /usr/share/keyrings/...`
exits 0 on an HTTP 404/5xx and writes the error body into the keyring file;
`set -eux` therefore sails past it, and the build only dies later at
`apt-get update` with an opaque GPG/NO_PUBKEY error instead of at the real
cause. Failure scenario: transient raw.githubusercontent.com outage or a
future relocation of ros.key → confusing signature-verification build failure
pointing at apt rather than the download. Found by line-audit of the RUN
body's failure modes (curl's non-`-f` semantics). The build can never produce
a silently-broken image (apt update always fails loudly), and the command
mirrors the official ROS 2 Humble docs verbatim — hence minor. Obvious fix:
`curl -fsSL`.

### Verdict

No blocker or major findings. The three files conform to Sec 2/Sec 9
line by line, and every mechanism they implement was executed successfully
end-to-end (compose config, image build, coexistence import, overlay
sourcing, fail-fast paths). One minor robustness nit (INFRA-1) is open.

## Master round 1

- **BUG MR1-INFRA-1 (major)** — Intermittent SIGSEGV (exit 139) during node
  startup when stale Fast DDS shared-memory segments from a SIGKILLed
  predecessor exist. Files: `compose.yaml` (`ipc: host` — segments persist on
  the host across containers) + `run.sh` (no cleanup/mitigation).
  Reproduction: after `docker kill ds-ros-pipeline` (Sec 10 test 8) or any
  `docker rm -f`, `/dev/shm/fastrtps_*` segments remain; a subsequent
  `docker compose -f ds_ros_pipeline/compose.yaml run -d ds-ros-pipeline
  bash -lc 'ds_ros_pipeline/run.sh --ros-args -p source.loop:=false'` then
  dies ~50% of the time with `Fatal Python error: Aborted` + `Segmentation
  fault` inside `rclpy` `Node.__init__` (traceback: ros_io.py:137 →
  ds_node.py:155 → ds_node.py:447; captured with PYTHONFAULTHANDLER=1).
  Controlled experiment on this machine (RTX 3070 host, 2026-07-20):
  4 launches with stale segments present → runs 2 and 4 exited 139 before
  "services up"; 4 launches each preceded by
  `rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*` → 4/4 started clean.
  Observed vs expected: relaunch after the crash-tolerance test is a coin
  flip vs deterministic bring-up. Suggested fix in run.sh: purge stale
  `fastrtps` segments owned by dead PIDs at startup, or disable the Fast DDS
  SHM transport (UDP-only profile is sufficient with network_mode: host).
  Side effect of the same root cause seen during testing: the in-container
  `ros2` CLI daemon can wedge with `!rclpy.ok()` after subscribers are
  SIGKILLed (recovered via `ros2 daemon stop/start`).

## Coder fix (master round 1)

**MR1-INFRA-1 — fixed in `run.sh`** (no Dockerfile/compose change, so the
image rebuild is fully cache-hit). After `source /opt/ros/humble/setup.bash`
and before anything else, run.sh now runs `fastdds shm clean || true`. The
`fastdds` CLI ships with ros-humble-ros-base (`/opt/ros/humble/bin/fastdds`,
verified in the built image); `shm clean` removes only *dead-owner* ports and
segments — liveness is checked via file locks on the `*_el` files, which work
across PID namespaces, so with `ipc: host` a live participant anywhere on the
host is never touched (unlike a blanket `rm -f /dev/shm/fastrtps_*`, which
races concurrent participants). Kept the SHM transport enabled rather than
forcing a UDP-only profile: /vlm_raw ships ~11 MB raw frames per message and
SHM/data-sharing is the transport the design's throughput expectations rest
on; the zombie purge removes the crash trigger without that regression.
`|| true` guarantees startup never blocks on the cleanup.

Evidence (2026-07-20, this host):

1. Root condition reproduced: an rclpy pub/sub pair exchanging 11 MB Image
   messages over SHM in an `--ipc=host` container, SIGKILLed mid-publish,
   leaves `fastrtps_*`/`sem.fastrtps_*` files in the **host** /dev/shm after
   the container exits (observed 7–9 stale files per kill).
2. Fix mechanism verified safe under concurrency: with 9 zombie files present
   AND a live participant running in another `--ipc=host` container,
   `fastdds shm clean` reported "1 ports in use / 1 segments in use", removed
   every zombie, and left the live participant's segment + in-use port intact.
3. End-to-end through run.sh: recreated zombies (7 files), then invoked
   `bash -lc ds_ros_pipeline/run.sh` in an `--ipc=host` container with the
   repo mounted at the compose workdir — startup log shows `shm.clean: … 1
   zombie ports cleaned / 2 zombie segments cleaned` and host /dev/shm is
   left with 0 fastrtps files, i.e. every subsequent launch starts from the
   exact state the tester's controlled experiment showed to be 4/4 clean.
   (6/6 probe node launches after cleanup exited 0 here as well.)
4. Fail-fast behavior unchanged after the edit: CDCL_ROS_SETUP unset → rc=1
   with the clear message; set to a missing file → rc=1 naming the path (the
   shm clean correctly runs before both checks, so cleanup happens even on a
   refused start).
5. Required image validation: `docker compose -f ds_ros_pipeline/compose.yaml
   build` succeeds (all layers cached); in the built image, one interpreter
   runs `import rclpy, pyds, gi; …Gst` → "ok" (`--gpus all`), and with the
   real overlay mounted ro at its build path, `import cdcl_umd_msgs.msg`
   exposes TargetBox/TargetBoxArray.

Not addressed here: the tester-noted `ros2` CLI daemon wedge after SIGKILLed
subscribers — same root cause, but it lives in the interactive `exec` shells,
not in these three files; `run.sh`'s startup purge also clears the zombies
that wedge it, and `ros2 daemon stop/start` remains the in-shell recovery.

## Master round 2

- **MR1-INFRA-1 — VERIFIED FIXED.** The `fastdds shm clean` added to
  `run.sh` engages and removes the crash trigger. Evidence (2026-07-20/21,
  this host): after Sec 10 test 8 (`docker kill ds-ros-pipeline` mid-
  recording) the host /dev/shm held 10 stale `fastrtps_*` files — the exact
  round-1 crash precondition; the immediate relaunch logged "2 zombie ports
  cleaned / 2 zombie segments cleaned" and reached "ds_node: pipelines
  PLAYING, services up". Three further kill→relaunch cycles all started
  clean (4/4 total vs ~50% exit-139 in round 1). No new infra findings:
  bring-up via `docker compose -f ds_ros_pipeline/compose.yaml up --build
  -d` was clean (image cached, engines deserialized in ~2 s), and the
  `run --rm ... -p source.loop:=false` variant (test 15) also started and
  ran the shm purge.
