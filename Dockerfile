FROM nvcr.io/nvidia/deepstream:9.0-samples-multiarch

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV CUDA_CACHE_DISABLE=0
ENV QT_X11_NO_MITSHM=1

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    bash-completion \
    vim \
    nano \
    less \
    gdb \
    sudo \
    gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user matching host UID/GID.
# If the base image already has UID/GID 1000, reuse/rename safely.
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

COPY .bashrc.container /home/${USERNAME}/.bashrc
RUN chown ${USER_UID}:${USER_GID} /home/${USERNAME}/.bashrc

WORKDIR /home/user/deepstream-work

USER ${USERNAME}

CMD ["/bin/bash"]
