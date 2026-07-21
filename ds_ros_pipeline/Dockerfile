# ROS2 layered on top of the DeepStream development image (DESIGN.md §2).
#
# The base DeepStream release and the ROS2 distro are build args because the
# pairing is fixed by Ubuntu: DS 7.1 is jammy, whose ROS2 LTS is Humble (the
# validated default); DS 9.0 is noble, whose ROS2 LTS is Jazzy (required on
# Blackwell-generation GPUs, whose sm_120 DS 7.1's TensorRT cannot build
# engines for). Either way rclpy and pyds share the base image's one Python
# (3.10 on 7.1, 3.12 on 9.0). The apt suite is read from the base image's own
# /etc/os-release rather than hardcoded, so the repo line always matches the
# base. The chosen distro is baked into DS_ROS_DISTRO for run.sh.
#
# The existing deepstream-work base image and every existing compose service
# are untouched; this file only adds layers on top.
#
# Layer ordering (project convention: new/fixable steps last, big stable steps
# first): ros-base is by far the largest download, so the repo setup +
# ros-base install stands alone in the first layer; the small add-on message
# packages and colcon sit in a separate later layer, so iterating on package
# additions or anything below never re-downloads ros-base.
#
# No cv_bridge: messages are built by hand from numpy (one less dependency);
# OpenCV-headless and numpy are already in the base image, and
# GIO_USE_PROXY_RESOLVER=dummy is inherited from the base image's ENV.
ARG DS_VERSION=7.1
FROM deepstream-work:${DS_VERSION}

USER root

# ARGs declared before FROM are consumed by FROM only; ROS_DISTRO is declared
# here so it is usable below.
ARG ROS_DISTRO=humble
ENV DS_ROS_DISTRO=${ROS_DISTRO}

# ROS2 apt repo (ros.key + the base image's own suite) and the one big install.
RUN set -eux; \
    codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"; \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${codename} main" \
        > /etc/apt/sources.list.d/ros2.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends "ros-${ROS_DISTRO}-ros-base"; \
    rm -rf /var/lib/apt/lists/*

# Message packages beyond ros-base (diagnostic_msgs ships with ros-base's
# common_interfaces; installing it explicitly is a no-op that documents the
# dependency), plus colcon so this image can build the cdcl_umd_msgs overlay
# for its own distro/Python when the host workspace was built for another
# (see compose.yaml's DS_CDCL_ROS_WS note). Separate layer so adding a
# package rebuilds only this step.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        "ros-${ROS_DISTRO}-diagnostic-msgs" \
        "ros-${ROS_DISTRO}-vision-msgs" \
        python3-colcon-common-extensions; \
    rm -rf /var/lib/apt/lists/*

USER user

CMD ["/bin/bash"]
