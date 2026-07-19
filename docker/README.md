# Build environment

`Dockerfile` builds the DeepStream development image. `Dockerfile.ros-humble`
builds the ROS Humble publisher image. Both are driven by `docker-compose.yml`
at the repo root, and both use the repo root as their build context, so paths
inside them are written relative to the root (`COPY docker/.bashrc.container`).

## Selecting a DeepStream release

The release is a build argument defaulting to 7.1:

```bash
scripts/build.sh                            # 7.1
DS_VERSION=9.0 scripts/build.sh             # 9.0
DS_VERSION=8.0 DS_IMAGE_FLAVOR=triton-multiarch scripts/build.sh
```

The image is tagged `deepstream-work:<DS_VERSION>` so releases do not overwrite
each other.

## What changes per release

Each DeepStream release pins its own Python interpreter and CUDA toolkit, and
NVIDIA's `pyds` wheels carry a CPython ABI tag locked to that interpreter. The
wheel and the base image therefore have to be chosen together.

| DeepStream | Ubuntu | Python | pyds | CUDA (for the YOLO parser) |
| --- | --- | --- | --- | --- |
| 7.1 | 22.04 | 3.10 | 1.2.0 wheel (cp310) | 12.6 |
| 8.0 | 24.04 | 3.12 | 1.2.2 wheel (cp312) | 12.8 |
| 9.0 | 24.04 | 3.12 | no wheel published; source build | 13.1 |

`scripts/build_yolo_parser.sh` derives the CUDA version from the DeepStream
version at runtime, so it does not need updating when the base image changes.
`CUDA_VERSION` overrides it. This matters because DeepStream-Yolo's Makefile
interpolates `CUDA_VER` directly into `/usr/local/cuda-$(CUDA_VER)/{include,lib64}`
— it must name a directory that exists, not merely a compatible toolkit.

To target a release not in the table, pass the wheel explicitly:

```bash
docker compose build --build-arg DS_VERSION=7.0 \
  --build-arg PYDS_WHEEL_URL=https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.11/pyds-1.1.11-py3-none-linux_x86_64.whl
```

With no wheel URL and no table entry, the build falls back to compiling the
bindings from source at `DEEPSTREAM_PYTHON_APPS_REF`.

## Layer order

Layers run cheapest-and-most-stable first, with the `pyds` install last. That
step is the one most likely to break on a new DeepStream release, and keeping it
at the end means a failure there leaves the apt and user-setup layers cached, so
retries are seconds rather than minutes. Keep it that way when editing.

## Two things that look like bugs but are not

The build verifies `pyds` with `importlib.util.find_spec` instead of importing
it. `pyds` links against `libcuda.so.1`, which the NVIDIA container runtime
injects at *run* time; it does not exist during a build, so a real import fails
even when the install is correct. The genuine import is covered by
`validation/smoke_pipeline.py`, which runs with the GPU attached.

There is no CUDA "compat" symlinking here, and it should not be added back. An
earlier revision linked `libcuda.so.<driver>` out of `/usr/local/cuda-*/compat`
into `lib64`, which inverts CUDA forward compatibility — that mechanism exists
to run a *newer* CUDA on an *older* driver, not to shadow a newer host driver
with an older stub. A host driver newer than the container's CUDA (for example
driver 580 with CUDA 12.6) is the normal, supported direction and needs no help.

## Verifying a build

```bash
docker compose build deepstream-dev
docker compose run --rm deepstream-dev python3 -c "import pyds; print(pyds.__file__)"
docker compose run --rm deepstream-dev scripts/build_yolo_parser.sh
docker compose run --rm deepstream-dev python3 validation/smoke_pipeline.py --frames 60
```
