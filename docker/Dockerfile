# DeepStream development image.
#
# The DeepStream release is selected at build time so the same Dockerfile can
# target 7.1 (default), 8.0, or 9.0:
#
#   docker compose build                      # 7.1
#   DS_VERSION=9.0 docker compose build       # 9.0
#
# Layers are ordered cheapest-and-most-stable first. The Python bindings
# install is deliberately LAST because it is the step most likely to fail on a
# new DeepStream release, and putting it at the end keeps every earlier layer
# cached across retries.
ARG DS_VERSION=7.1
ARG DS_IMAGE_FLAVOR=samples-multiarch

FROM nvcr.io/nvidia/deepstream:${DS_VERSION}-${DS_IMAGE_FLAVOR}

# ARGs declared before FROM are consumed by FROM only; re-declare to use below.
ARG DS_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_CACHE_DISABLE=0
ENV QT_X11_NO_MITSHM=1
# Consumed by scripts/build_yolo_parser.sh to pick the matching CUDA toolkit.
ENV DS_VERSION=${DS_VERSION}

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000
ARG WORKSPACE_DIR=/workspace/deepstream-work

# Overrides the per-release pyds wheel resolved below. Set this when targeting a
# DeepStream release this Dockerfile does not know about yet.
ARG PYDS_WHEEL_URL=
# Only used by the source-build fallback (releases with no published wheel).
ARG DEEPSTREAM_PYTHON_APPS_REF=8ad0349ed7a496fae35ebb21c350641727070b89

# ---------------------------------------------------------------------------
# System packages. Verified present on both Ubuntu 22.04 (DS 7.1) and 24.04
# (DS 8.0/9.0), so this layer is shared across every supported release.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    gdb \
    git \
    graphviz \
    less \
    nano \
    pkg-config \
    sudo \
    vim \
    x11-apps \
    \
    python3-dev \
    python3-full \
    python3-gi \
    python3-gi-cairo \
    python3-pip \
    python3-venv \
    pybind11-dev \
    \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gst-rtsp-server-1.0 \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    libgirepository1.0-dev \
    libglib2.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Unprivileged user matching the host UID/GID so bind-mounted files stay
# writable from both sides.
# ---------------------------------------------------------------------------
RUN set -eux; \
    if getent group "${USER_GID}" >/dev/null; then \
        EXISTING_GROUP="$(getent group "${USER_GID}" | cut -d: -f1)"; \
    else \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
        EXISTING_GROUP="${USERNAME}"; \
    fi; \
    if id -u "${USER_UID}" >/dev/null 2>&1; then \
        EXISTING_USER="$(getent passwd "${USER_UID}" | cut -d: -f1)"; \
        usermod -l "${USERNAME}" "${EXISTING_USER}" || true; \
        usermod -d "/home/${USERNAME}" -m "${USERNAME}" || true; \
        usermod -g "${EXISTING_GROUP}" "${USERNAME}" || true; \
    else \
        useradd --uid "${USER_UID}" --gid "${EXISTING_GROUP}" -m "${USERNAME}" --shell /bin/bash; \
    fi; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"; \
    mkdir -p "/home/${USERNAME}"; \
    mkdir -p "${WORKSPACE_DIR}"; \
    chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}" "${WORKSPACE_DIR}"

COPY docker/.bashrc.container /home/${USERNAME}/.bashrc
RUN chown ${USER_UID}:${USER_GID} /home/${USERNAME}/.bashrc

WORKDIR ${WORKSPACE_DIR}

# ---------------------------------------------------------------------------
# DeepStream Python bindings (pyds). LAST because it is the most fragile step.
#
# NVIDIA publishes wheels whose ABI tag is locked to the interpreter shipped by
# the matching DeepStream base image, so the wheel and the base image must be
# chosen together:
#
#   DS 7.1 -> pyds 1.2.0, cp310, Ubuntu 22.04
#   DS 8.0 -> pyds 1.2.2, cp312, Ubuntu 24.04
#   DS 9.0 -> no wheel published; falls back to a source build
#
# Installing the wheel (rather than copying the raw pyds*.so into
# site-packages) means a wrong-ABI artifact fails loudly here at build time
# instead of surfacing as an ImportError at runtime.
# ---------------------------------------------------------------------------
RUN set -eux; \
    ARCH="$(uname -m)"; \
    PYDS_URL="${PYDS_WHEEL_URL}"; \
    if [ -z "${PYDS_URL}" ]; then \
        REL_BASE="https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download"; \
        case "${DS_VERSION}" in \
            7.1*) PYDS_URL="${REL_BASE}/v1.2.0/pyds-1.2.0-cp310-cp310-linux_${ARCH}.whl" ;; \
            8.0*) PYDS_URL="${REL_BASE}/v1.2.2/pyds-1.2.2-cp312-cp312-linux_${ARCH}.whl" ;; \
            *)    PYDS_URL="" ;; \
        esac; \
    fi; \
    PIP_FLAGS=""; \
    if pip3 install --help 2>/dev/null | grep -q -- --break-system-packages; then \
        PIP_FLAGS="--break-system-packages"; \
    fi; \
    if [ -n "${PYDS_URL}" ]; then \
        echo "Installing prebuilt pyds wheel: ${PYDS_URL}"; \
        pip3 install --no-cache-dir ${PIP_FLAGS} "${PYDS_URL}"; \
    else \
        echo "No published pyds wheel for DeepStream ${DS_VERSION}; building from source"; \
        git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git /opt/deepstream_python_apps; \
        cd /opt/deepstream_python_apps; \
        git checkout "${DEEPSTREAM_PYTHON_APPS_REF}"; \
        git submodule update --init; \
        cd bindings; \
        mkdir -p build; \
        cd build; \
        cmake ..; \
        make -j"$(nproc)"; \
        pip3 install --no-cache-dir ${PIP_FLAGS} ./pyds-*.whl \
            || { PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
                 cp pyds*.so "${PY_SITE}/"; }; \
    fi; \
    python3 -c "import pyds; print('pyds OK:', pyds.__file__)"

USER ${USERNAME}

CMD ["/bin/bash"]
