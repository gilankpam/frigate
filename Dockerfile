# syntax=docker/dockerfile:1.2

# https://askubuntu.com/questions/972516/debian-frontend-environment-variable
ARG DEBIAN_FRONTEND=noninteractive

FROM ubuntu:22.04 AS base

RUN apt update && \
    apt install -y ca-certificates && \
    sed -i 's/http:\/\/ports.ubuntu.com\/ubuntu-ports\//https:\/\/mirror.kumi.systems\/ubuntu-ports\//g' /etc/apt/sources.list

FROM base AS wget
ARG DEBIAN_FRONTEND
RUN apt-get update \
    && apt-get install -y wget xz-utils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /rootfs

FROM base AS nginx
ARG DEBIAN_FRONTEND

# bind /var/cache/apt to tmpfs to speed up nginx build
RUN --mount=type=tmpfs,target=/tmp --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=bind,source=docker/build_nginx.sh,target=/deps/build_nginx.sh \
    /deps/build_nginx.sh

FROM wget AS go2rtc
ARG TARGETARCH
WORKDIR /rootfs/usr/local/go2rtc/bin
RUN wget -qO go2rtc "https://github.com/AlexxIT/go2rtc/releases/download/v1.2.0/go2rtc_linux_${TARGETARCH}" \
    && chmod +x go2rtc

FROM wget AS models
COPY labelmap.txt .

FROM wget AS s6-overlay
ARG TARGETARCH
RUN --mount=type=bind,source=docker/install_s6_overlay.sh,target=/deps/install_s6_overlay.sh \
    /deps/install_s6_overlay.sh

FROM wget as librknn
# Download lib file v1.5.0
RUN mkdir -p usr/lib && \
    wget -O usr/lib/librknnrt.so https://github.com/rockchip-linux/rknpu2/raw/4e2101af40a6cbe253eb680673846b7ec832a896/runtime/RK3588/Linux/librknn_api/aarch64/librknnrt.so && \
    wget -O usr/lib/librknn_api.so https://github.com/rockchip-linux/rknpu2/raw/4e2101af40a6cbe253eb680673846b7ec832a896/runtime/RK3588/Linux/librknn_api/aarch64/librknn_api.so


FROM base AS wheels
ARG DEBIAN_FRONTEND
ARG TARGETARCH

# Use a separate container to build wheels to prevent build dependencies in final image
RUN apt-get -qq update \
    && apt-get -qq install -y \
    python3 \
    python3-dev \
    python3-pip \
    # opencv dependencies
    build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev libssl-dev\
    libtbb2 libtbb-dev libopenexr-dev \
    libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev \
    # scipy dependencies
    gcc gfortran libopenblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# RUN wget -q https://bootstrap.pypa.io/get-pip.py -O get-pip.py \
#     && python3 get-pip.py "pip"

RUN if [ "${TARGETARCH}" = "arm" ]; \
    then echo "[global]" > /etc/pip.conf \
    && echo "extra-index-url=https://www.piwheels.org/simple" >> /etc/pip.conf; \
    fi

COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

COPY requirements-wheels.txt /requirements-wheels.txt
RUN pip3 wheel --wheel-dir=/wheels -r requirements-wheels.txt


# Collect deps in a single layer
FROM scratch AS deps-rootfs
COPY --from=nginx /usr/local/nginx/ /usr/local/nginx/
COPY --from=go2rtc /rootfs/ /
COPY --from=s6-overlay /rootfs/ /
COPY --from=models /rootfs/ /
COPY --from=librknn /rootfs/ /
COPY docker/rootfs/ /


# Frigate deps (ffmpeg, python, nginx, go2rtc, s6-overlay, etc)
FROM base AS deps
ARG TARGETARCH

ARG DEBIAN_FRONTEND
# http://stackoverflow.com/questions/48162574/ddg#49462622
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn

ENV PATH="/usr/lib/btbn-ffmpeg/bin:/usr/local/go2rtc/bin:/usr/local/nginx/sbin:${PATH}"

# Install dependencies
RUN --mount=type=bind,source=docker/install_deps.sh,target=/deps/install_deps.sh \
    /deps/install_deps.sh

RUN --mount=type=bind,from=wheels,source=/wheels,target=/deps/wheels \
    pip3 install -U /deps/wheels/*.whl

COPY --from=deps-rootfs / /

RUN ldconfig

EXPOSE 5000
EXPOSE 1935
EXPOSE 8554
EXPOSE 8555/tcp 8555/udp

# Configure logging to prepend timestamps, log to stdout, keep 0 archives and rotate on 10MB
ENV S6_LOGGING_SCRIPT="T 1 n0 s10000000 T"

ENTRYPOINT ["/init"]
CMD []

# Frigate deps with Node.js and NPM for devcontainer
FROM deps AS devcontainer

# Do not start the actual Frigate service on devcontainer as it will be started by VSCode
# But start a fake service for simulating the logs
COPY docker/fake_frigate_run /etc/s6-overlay/s6-rc.d/frigate/run

# Create symbolic link to the frigate source code, as go2rtc's create_config.sh uses it
RUN mkdir -p /opt/frigate \
    && ln -svf /workspace/frigate/frigate /opt/frigate/frigate

# Install Node 16
RUN apt-get update \
    && apt-get install wget -y \
    && wget -qO- https://deb.nodesource.com/setup_16.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g npm@9

WORKDIR /workspace/frigate

RUN apt-get update \
    && apt-get install make -y \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=./requirements-dev.txt,target=/workspace/frigate/requirements-dev.txt \
    pip3 install -r requirements-dev.txt

CMD ["sleep", "infinity"]


# Frigate web build
FROM node:16 AS web-build

WORKDIR /work
COPY web/package.json web/package-lock.json ./
RUN npm install

COPY web/ ./
RUN npm run build \
    && mv dist/BASE_PATH/monacoeditorwork/* dist/assets/ \
    && rm -rf dist/BASE_PATH

# Collect final files in a single layer
FROM scratch AS rootfs

WORKDIR /opt/frigate/
COPY frigate frigate/
COPY migrations migrations/
COPY --from=web-build /work/dist/ web/

# Frigate final container
FROM deps AS frigate

WORKDIR /opt/frigate/
COPY --from=rootfs / /

# Frigate w/ TensorRT Support as separate image
FROM frigate AS frigate-tensorrt
RUN --mount=type=bind,from=trt-wheels,source=/trt-wheels,target=/deps/trt-wheels \
    pip3 install -U /deps/trt-wheels/*.whl && \
    ln -s libnvrtc.so.11.2 /usr/local/lib/python3.9/dist-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so && \
    ldconfig

# Dev Container w/ TRT
FROM devcontainer AS devcontainer-trt

RUN --mount=type=bind,from=trt-wheels,source=/trt-wheels,target=/deps/trt-wheels \
    pip3 install -U /deps/trt-wheels/*.whl
