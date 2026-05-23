FROM nvcr.io/nvidia/deepstream:8.0-samples-multiarch

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_CACHE_DISABLE=0
ENV QT_X11_NO_MITSHM=1

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

# ---------------------------------------------------------------------
# Base dev tools, Python, GStreamer, and DeepStream app dependencies
# ---------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
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
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-rtsp-server-1.0 \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-rtsp \
    gstreamer1.0-tools \
    libgirepository1.0-dev \
    libglib2.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# Build/install NVIDIA DeepStream Python bindings so /usr/bin/python3
# can import pyds.
# ---------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git /opt/deepstream_python_apps \
    && cd /opt/deepstream_python_apps \
    && git submodule update --init \
    && cd bindings \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make -j"$(nproc)" \
    && PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && cp pyds*.so "$PY_SITE/"

# ---------------------------------------------------------------------
# CUDA library compatibility.
# Some DeepStream containers expose CUDA compatibility libraries in
# /usr/local/cuda*/compat but not under the regular lib64 path.
# ---------------------------------------------------------------------
RUN set -eux; \
    for d in /usr/local/cuda*/compat; do \
        [ -d "$d" ] || continue; \
        parent="$(dirname "$d")"; \
        mkdir -p "$parent/lib64"; \
        for so in "$d"/*.so*; do \
            [ -e "$so" ] || continue; \
            ln -sf "$so" "$parent/lib64/$(basename "$so")"; \
        done; \
    done

# ---------------------------------------------------------------------
# User setup
# ---------------------------------------------------------------------
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
    chown -R "${USER_UID}:${USER_GID}" "/home/${USERNAME}"

# ---------------------------------------------------------------------
# CUDA dev layout compatibility for DeepStream-Yolo Makefile.
# Build parser with CUDA_VER=12.5:
# - CUDA 12.5 has complete headers/runtime.
# - CUDA 12.8 has cuBLAS in this image.
# ---------------------------------------------------------------------
RUN set -eux; \
    rm -rf /usr/local/cuda-12.5/include /usr/local/cuda-12.5/lib64; \
    ln -s /usr/local/cuda-12.5/targets/x86_64-linux/include /usr/local/cuda-12.5/include; \
    ln -s /usr/local/cuda-12.5/targets/x86_64-linux/lib /usr/local/cuda-12.5/lib64; \
    ln -sf /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12 /usr/local/cuda-12.5/lib64/libcublas.so.12; \
    ln -sf /usr/local/cuda-12.5/lib64/libcublas.so.12 /usr/local/cuda-12.5/lib64/libcublas.so; \
    test -e /usr/local/cuda-12.5/include/cuda_runtime_api.h; \
    test -e /usr/local/cuda-12.5/include/crt/host_defines.h; \
    test -e /usr/local/cuda-12.5/lib64/libcublas.so

COPY .bashrc.container /home/${USERNAME}/.bashrc
RUN chown ${USER_UID}:${USER_GID} /home/${USERNAME}/.bashrc

WORKDIR /home/user/deepstream-work

USER ${USERNAME}

CMD ["/bin/bash"]
