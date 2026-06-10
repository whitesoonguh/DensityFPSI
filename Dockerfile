FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates build-essential cmake ninja-build git libtool \
    pkg-config python3 nasm libssl-dev libgmp-dev wget \
    && update-ca-certificates && rm -rf /var/lib/apt/lists/*

COPY . /home/DensityFPSI
WORKDIR /home/DensityFPSI

# 1. fetch + build libOTe into thirdparty/out/install
RUN bash thirdparty/build_deps.sh

# 2. configure + build the project against the real OT backend
RUN cmake --preset release -DSOOPRF_WITH_LIBOTE=ON \
    && cmake --build --preset release -j
