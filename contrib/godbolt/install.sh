#!/usr/bin/env bash
#
# install.sh - Install the rust-cuda toolchain for Compiler Explorer.
#
# This script is meant to be run on a CE builder node (or locally for
# testing).  It performs the following:
#
#   1. Installs the pinned Rust nightly with required components.
#   2. Clones the rust-cuda repository (or uses a local checkout).
#   3. Builds the rustc_codegen_nvvm codegen backend against libnvvm.
#   4. Copies the backend, cuda_std sources, and the wrapper script into
#      a self-contained prefix under /opt/compiler-explorer/rust-cuda/.
#
# Prerequisites:
#   - CUDA toolkit installed (CUDA_PATH or /usr/local/cuda)
#   - cmake, ninja-build, clang, pkg-config, libssl-dev, zlib1g-dev
#   - For LLVM 7 path: the prebuilt LLVM archive is downloaded automatically
#     by the codegen's build.rs, or you can pre-install LLVM 7 and export
#     LLVM_CONFIG=/path/to/llvm-config-7.
#
# Environment variables:
#   INSTALL_PREFIX   - Where to install (default: /opt/compiler-explorer/rust-cuda)
#   CUDA_PATH        - CUDA toolkit root  (default: /usr/local/cuda)
#   RUST_CUDA_REPO   - Path to an existing rust-cuda checkout (skips git clone)
#   RUST_CUDA_REF    - Git ref to check out (default: main)

set -euo pipefail

INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/compiler-explorer/rust-cuda}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
RUST_CUDA_REF="${RUST_CUDA_REF:-main}"

NIGHTLY="nightly-2026-04-02"
COMPONENTS="rust-src,rustc-dev,llvm-tools-preview"

echo "==> rust-cuda Compiler Explorer installer"
echo "    prefix:  ${INSTALL_PREFIX}"
echo "    CUDA:    ${CUDA_PATH}"
echo "    nightly: ${NIGHTLY}"

# ---------------------------------------------------------------------------
# 1. Install the pinned Rust nightly
# ---------------------------------------------------------------------------
echo "==> Installing Rust ${NIGHTLY} ..."
if ! command -v rustup &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
    export PATH="${HOME}/.cargo/bin:${PATH}"
fi

rustup toolchain install "${NIGHTLY}" --component "${COMPONENTS}"
rustup default "${NIGHTLY}"

echo "    rustc: $(rustc --version)"

# ---------------------------------------------------------------------------
# 2. Get the rust-cuda source
# ---------------------------------------------------------------------------
if [[ -n "${RUST_CUDA_REPO:-}" ]]; then
    REPO_DIR="${RUST_CUDA_REPO}"
    echo "==> Using existing checkout at ${REPO_DIR}"
else
    REPO_DIR="$(mktemp -d -t rust-cuda-src.XXXXXXXXXX)"
    echo "==> Cloning rust-cuda (ref: ${RUST_CUDA_REF}) into ${REPO_DIR} ..."
    git clone --depth 1 --branch "${RUST_CUDA_REF}" \
        https://github.com/Rust-GPU/rust-cuda.git "${REPO_DIR}"
fi

# ---------------------------------------------------------------------------
# 3. Build the codegen backend
# ---------------------------------------------------------------------------
echo "==> Building rustc_codegen_nvvm ..."
export LD_LIBRARY_PATH="${CUDA_PATH}/nvvm/lib64:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"

cd "${REPO_DIR}"
cargo build -p rustc_codegen_nvvm --release

# Find the built .so
CODEGEN_SO="$(find target/release -maxdepth 2 -name 'librustc_codegen_nvvm.so' -print -quit 2>/dev/null || true)"
if [[ -z "${CODEGEN_SO}" ]]; then
    # Try the deps directory with hash suffix.
    CODEGEN_SO="$(find target/release/deps -maxdepth 1 -name 'librustc_codegen_nvvm-*.so' -print -quit 2>/dev/null || true)"
fi
if [[ -z "${CODEGEN_SO}" ]]; then
    echo "error: could not find librustc_codegen_nvvm.so after build" >&2
    exit 1
fi
echo "    codegen backend: ${CODEGEN_SO}"

# ---------------------------------------------------------------------------
# 4. Install into the prefix
# ---------------------------------------------------------------------------
echo "==> Installing to ${INSTALL_PREFIX} ..."
mkdir -p "${INSTALL_PREFIX}"/{bin,lib,crates}

# Backend shared library.
cp "${CODEGEN_SO}" "${INSTALL_PREFIX}/lib/librustc_codegen_nvvm.so"

# Copy the crates that kernel code depends on at build time.
for crate in cuda_std cuda_std_macros; do
    cp -a "${REPO_DIR}/crates/${crate}" "${INSTALL_PREFIX}/crates/${crate}"
done

# Copy workspace-level files needed by cargo (Cargo.lock is especially
# important so dependency resolution is reproducible).
cp "${REPO_DIR}/Cargo.lock" "${INSTALL_PREFIX}/" 2>/dev/null || true

# Build and install the wrapper binary.
(
    cd "${REPO_DIR}/contrib/godbolt/rust-cuda-wrapper"
    cargo build --release
)
cp "${REPO_DIR}/contrib/godbolt/rust-cuda-wrapper/target/release/rust-cuda-wrapper" \
    "${INSTALL_PREFIX}/bin/"
chmod +x "${INSTALL_PREFIX}/bin/rust-cuda-wrapper"

# Version marker.
echo "${NIGHTLY}" > "${INSTALL_PREFIX}/rust-toolchain-version"

# Also copy any native libs the codegen may need at link time (from the
# same build directory).
for lib in "${REPO_DIR}"/target/release/deps/lib*.so; do
    [[ -f "${lib}" ]] && cp "${lib}" "${INSTALL_PREFIX}/lib/" 2>/dev/null || true
done

echo "==> Installation complete."
echo ""
echo "Test with:"
echo "  RUST_CUDA_ROOT=${INSTALL_PREFIX} CUDA_PATH=${CUDA_PATH} \\"
echo "    ${INSTALL_PREFIX}/bin/rust-cuda-wrapper contrib/godbolt/test-kernel.rs"
