FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates build-essential cmake ninja-build git libtool \
    autoconf automake pkg-config python3 nasm libssl-dev libgmp-dev wget \
    && update-ca-certificates && rm -rf /var/lib/apt/lists/*

COPY . /home/DensityFPSI
WORKDIR /home/DensityFPSI

# AVX-512 toggle. Default OFF so the image runs on any x86-64 with AVX2 + VAES +
# VPCLMUL (e.g. AMD Zen3). For the paper's AVX-512 reference profile build with:
#   docker build --build-arg ENABLE_AVX512=ON -t densityfpsi .
# NOTE: the build host's CPU must actually support what you enable; the CMake
# check only verifies the *compiler* accepts the flags, not the *CPU*, so an
# AVX-512 build run on a non-AVX-512 CPU will fault with "Illegal instruction".
ARG ENABLE_AVX512=OFF

# 1. fetch + build libOTe into thirdparty/out/install
RUN bash thirdparty/build_deps.sh

# 2. configure + build the project against the real OT backend.
#    Explicit -B build (not a preset) so the output path is always build/.
RUN cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DSOOPRF_WITH_LIBOTE=ON \
      -DFPSI_ENABLE_NATIVE=ON \
      -DFPSI_ENABLE_AVX512=${ENABLE_AVX512} \
      -DFPSI_REQUIRE_AVX512_VAES_VPCLMUL=${ENABLE_AVX512} \
    && cmake --build build -j
