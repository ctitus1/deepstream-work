FROM nvcr.io/nvidia/deepstream:9.0-samples-multiarch

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_CACHE_DISABLE=0
ENV QT_X11_NO_MITSHM=1

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000
ARG WORKSPACE_DIR=/workspace/deepstream-work
ARG DEEPSTREAM_PYTHON_APPS_REF=8ad0349ed7a496fae35ebb21c350641727070b89

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

# Build/install NVIDIA DeepStream Python bindings so /usr/bin/python3 can import pyds.
RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git /opt/deepstream_python_apps \
    && cd /opt/deepstream_python_apps \
    && git checkout "${DEEPSTREAM_PYTHON_APPS_REF}" \
    && git submodule update --init \
    && cd bindings \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make -j"$(nproc)" \
    && PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && cp pyds*.so "$PY_SITE/"

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

COPY .bashrc.container /home/${USERNAME}/.bashrc
RUN chown ${USER_UID}:${USER_GID} /home/${USERNAME}/.bashrc

WORKDIR ${WORKSPACE_DIR}

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    gir1.2-gst-rtsp-server-1.0 \
    && rm -rf /var/lib/apt/lists/*

USER ${USERNAME}

CMD ["/bin/bash"]
