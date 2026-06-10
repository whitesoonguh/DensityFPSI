#!/usr/bin/env bash
# Fetch + build libOTe as a local third-party dependency, mirroring the
# Th0masAndy/FPSI layout: everything is installed under thirdparty/out/install
# and consumed by the main build via find_package(libOTe).
#
# Usage (from anywhere):   bash thirdparty/build_deps.sh
# Then configure the project with -DSOOPRF_WITH_LIBOTE=ON (see README below).
#
# System packages required once (Ubuntu 24.04):
#   sudo apt-get install -y build-essential cmake git libtool python3 nasm \
#       libssl-dev libgmp-dev wget pkg-config
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
THIRD="$ROOT/thirdparty"
PREFIX="$THIRD/out/install"

# Pin the SAME libOTe commit that secure-join (the AltMod-WPRF reference) builds
# against. It is known to expose MultType::Tungsten + Silent OT/VOLE in the
# shape altmod_ot_gen.cpp uses. Override with: LIBOTE_TAG=<sha> bash build_deps.sh
LIBOTE_TAG="${LIBOTE_TAG:-d21bc4d7aae941e276b92615252fd1760c902890}"

mkdir -p "$THIRD"
cd "$THIRD"

if [ ! -d libOTe ]; then
    git clone https://github.com/osu-crypto/libOTe.git
fi
cd libOTe
git fetch --all --tags --quiet || true
git checkout "$LIBOTE_TAG"

# libOTe's build.py forwards to cryptoTools.build:
#   --install=PATH   install headers + libs + libOTeConfig.cmake to PATH
#   --boost --sodium fetch+build those dependencies locally
#   -D...            forwarded CMake cache vars
# We enable the silent primitives the AltMod OT tape needs.
python3 build.py \
    --install="$PREFIX" \
    --boost --sodium \
    -DENABLE_SILENTOT=ON \
    -DENABLE_SILENT_VOLE=ON \
    -DENABLE_SOFTSPOKEN_OT=ON \
    -DSODIUM_MONTGOMERY=false

echo
echo "libOTe installed to: $PREFIX"
echo "Now configure with:  cmake --preset release -DSOOPRF_WITH_LIBOTE=ON"