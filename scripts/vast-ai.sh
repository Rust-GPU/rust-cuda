#!/bin/bash

# Build vecadd on the Nix-equipped build host, then push it to a vast.ai
# Blackwell box with CUDA 13.2+ and run it there.

set -euo pipefail

BUILD_HOST="${BUILD_HOST:-brandon@asusrogstrix.local}"
BUILD_DIR="${BUILD_DIR:-/home/brandon/Rust-CUDA}"
BUILD_BIN="${BUILD_BIN:-target/debug/vecadd}"

VAST_HOST="${VAST_HOST:-root@ssh6.vast.ai}"
VAST_PORT="${VAST_PORT:-34929}"
VAST_DEST="${VAST_DEST:-/workspace/vecadd}"

# Vast.ai hands us a new container (→ new host key) on every rental, so skip
# the TOFU prompt and keep the churn out of ~/.ssh/known_hosts.
VAST_SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)

# Nix-built binaries bake /nix/store/...-glibc/ld-linux-x86-64.so.2 as their ELF
# interpreter, which doesn't exist on the vast.ai container. Rewrite it to the
# standard FHS path at build time and strip the rpath so the loader only pulls
# in the container's glibc + the CUDA driver's libcuda.so.1.
echo ">> Building on $BUILD_HOST"
ssh "$BUILD_HOST" "cd '$BUILD_DIR' \
  && nix develop .#v19 --command cargo build -p vecadd \
  && nix shell nixpkgs#patchelf --command patchelf \
       --set-interpreter /lib64/ld-linux-x86-64.so.2 \
       --remove-rpath '$BUILD_BIN'"

echo ">> Staging binary locally"
local_bin="$(mktemp -d)/vecadd"
scp "$BUILD_HOST:$BUILD_DIR/$BUILD_BIN" "$local_bin"

echo ">> Uploading to $VAST_HOST:$VAST_PORT"
scp "${VAST_SSH_OPTS[@]}" -P "$VAST_PORT" "$local_bin" "$VAST_HOST:$VAST_DEST"

echo ">> Running on vast.ai"
ssh "${VAST_SSH_OPTS[@]}" -p "$VAST_PORT" "$VAST_HOST" "chmod +x '$VAST_DEST' && nvidia-smi -L && '$VAST_DEST'"
