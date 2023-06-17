#!/bin/bash

set -euxo pipefail

apt-get -qq update

apt-get -qq install --no-install-recommends -y \
    apt-transport-https \
    gnupg \
    wget \
    procps vainfo \
    unzip locales tzdata libxml2 xz-utils \
    python3-pip \
    curl \
    jq \
    software-properties-common

mkdir -p -m 600 /root/.gnupg

# if [[ "${TARGETARCH}" == "arm64" ]]; then
#     apt-get -qq install --no-install-recommends --no-install-suggests -y \
#         libva-drm2 mesa-va-drivers
# fi

add-apt-repository -y ppa:liujianfeng1994/panfork-mesa
add-apt-repository -y ppa:liujianfeng1994/rockchip-multimedia

apt install -y mali-g610-firmware ocl-icd-opencl-dev libmali-g610-x11 ffmpeg

apt-get purge gnupg apt-transport-https wget xz-utils -y
apt-get clean autoclean -y
apt-get autoremove --purge -y
rm -rf /var/lib/apt/lists/*
