# ROS2 Humble layered on top of the DeepStream development image (DESIGN.md §2).
#
# DS 7.1 is Ubuntu 22.04 (jammy) -- exactly the distro ROS2 Humble targets --
# so rclpy and pyds 1.2.0 share one Python 3.10 interpreter. The existing
# deepstream-work:7.1 image and every existing compose service are untouched;
# this file only adds layers on top of it.
#
# Layer ordering (project convention: new/fixable steps last, big stable steps
# first): ros-humble-ros-base is by far the largest download, so the repo
# setup + ros-base install stands alone in the first layer; the small add-on
# message packages sit in a separate later layer, so iterating on package
# additions or anything below never re-downloads ros-base.
#
# No cv_bridge: messages are built by hand from numpy (one less dependency);
# OpenCV-headless and numpy are already in the base image, and
# GIO_USE_PROXY_RESOLVER=dummy is inherited from the base image's ENV.
FROM deepstream-work:7.1

USER root

# ROS2 apt repo (ros.key + jammy main) and the one big install.
RUN set -eux; \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
        > /etc/apt/sources.list.d/ros2.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends ros-humble-ros-base; \
    rm -rf /var/lib/apt/lists/*

# Message packages beyond ros-base (diagnostic_msgs ships with ros-base's
# common_interfaces; installing it explicitly is a no-op that documents the
# dependency). Separate layer so adding a package rebuilds only this step.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ros-humble-diagnostic-msgs \
        ros-humble-vision-msgs; \
    rm -rf /var/lib/apt/lists/*

USER user

CMD ["/bin/bash"]
