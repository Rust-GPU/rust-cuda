#!/usr/bin/env bash
#
# rust-cuda-wrapper.sh - Compiler Explorer wrapper for rust-cuda.
#
# Godbolt invokes this as the "compiler binary". It accepts a single .rs file
# containing a #[kernel] GPU function, wraps it in a temporary Cargo project
# that depends on cuda_std, builds it with rustc_codegen_nvvm targeting
# nvptx64-nvidia-cuda, and emits the resulting PTX (or LLVM IR) on stdout.
#
# Environment expected to be pre-configured by install.sh:
#   RUST_CUDA_ROOT  - /opt/compiler-explorer/rust-cuda
#   CUDA_PATH       - CUDA toolkit root (e.g. /usr/local/cuda)
#
# Usage:
#   rust-cuda-wrapper.sh [flags] <input.rs>
#
# Flags:
#   --emit=ptx          Output PTX assembly (default)
#   --emit=llvm-ir      Output LLVM IR before libnvvm conversion
#   --opt-level=N       Optimisation level: 0 or 3 (default 3)
#   --gpu-arch=smXX     Target GPU arch, e.g. sm_75 (default sm_75)
#   --version           Print version info and exit

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RUST_CUDA_ROOT="${RUST_CUDA_ROOT:-/opt/compiler-explorer/rust-cuda}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"

EMIT="ptx"
OPT_LEVEL="3"
# Default to compute_75 (Turing), matching NvvmArch::default() when llvm19 is off.
GPU_ARCH="compute_75"
INPUT_FILE=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --emit=*)
            EMIT="${1#--emit=}"
            shift
            ;;
        --opt-level=*)
            OPT_LEVEL="${1#--opt-level=}"
            shift
            ;;
        --gpu-arch=*)
            raw="${1#--gpu-arch=}"
            # Accept sm_XX shorthand and convert to compute_XX.
            GPU_ARCH="${raw/sm_/compute_}"
            shift
            ;;
        --version)
            echo "rust-cuda-wrapper for Compiler Explorer"
            echo "Toolchain: $(cat "${RUST_CUDA_ROOT}/rust-toolchain-version" 2>/dev/null || echo unknown)"
            echo "CUDA: $(${CUDA_PATH}/bin/nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' || echo unknown)"
            exit 0
            ;;
        -*)
            # Silently ignore other flags Godbolt may pass (e.g. -o, -S).
            shift
            ;;
        *)
            INPUT_FILE="$1"
            shift
            ;;
    esac
done

if [[ -z "${INPUT_FILE}" ]]; then
    echo "error: no input file" >&2
    exit 1
fi

if [[ ! -f "${INPUT_FILE}" ]]; then
    echo "error: input file '${INPUT_FILE}' not found" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Create a temporary Cargo project
# ---------------------------------------------------------------------------
WORK_DIR="$(mktemp -d -t rust-cuda-godbolt.XXXXXXXXXX)"
cleanup() { rm -rf "${WORK_DIR}"; }
trap cleanup EXIT

CRATE_DIR="${WORK_DIR}/gpu_kernel"
mkdir -p "${CRATE_DIR}/src"

# Cargo.toml for the kernel crate.
cat > "${CRATE_DIR}/Cargo.toml" <<'CARGO_EOF'
[package]
name = "gpu_kernel"
version = "0.1.0"
edition = "2024"

[dependencies]
cuda_std = { path = "__CUDA_STD_PATH__" }

[lib]
crate-type = ["cdylib", "rlib"]
CARGO_EOF

sed -i "s|__CUDA_STD_PATH__|${RUST_CUDA_ROOT}/crates/cuda_std|" "${CRATE_DIR}/Cargo.toml"

# Copy the user's source file as src/lib.rs.
cp "${INPUT_FILE}" "${CRATE_DIR}/src/lib.rs"

# ---------------------------------------------------------------------------
# Locate the codegen backend
# ---------------------------------------------------------------------------
CODEGEN_SO="${RUST_CUDA_ROOT}/lib/librustc_codegen_nvvm.so"
if [[ ! -f "${CODEGEN_SO}" ]]; then
    echo "error: codegen backend not found at ${CODEGEN_SO}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build RUSTFLAGS - mirrors cuda_builder's invoke_rustc()
# ---------------------------------------------------------------------------
RUSTFLAGS_ARRAY=(
    "-Zcodegen-backend=${CODEGEN_SO}"
    "-Zunstable-options"
    "-Zcrate-attr=feature(register_tool)"
    "-Zcrate-attr=register_tool(nvvm_internal)"
    "-Zcrate-attr=no_std"
    "-Zsaturating_float_casts=false"
    "-Cpanic=immediate-abort"
)

# LLVM / libnvvm arguments
LLVM_ARGS="-arch=${GPU_ARCH}"
LLVM_ARGS+=" --override-libm"

if [[ "${OPT_LEVEL}" == "0" ]]; then
    LLVM_ARGS+=" -opt=0"
fi

# Emit mode
if [[ "${EMIT}" == "llvm-ir" ]]; then
    RUSTFLAGS_ARRAY+=("--emit=llvm-ir")
fi

RUSTFLAGS_ARRAY+=("-Cllvm-args=${LLVM_ARGS}")

# Join with unit separator (\x1f), the same encoding cargo uses for
# CARGO_ENCODED_RUSTFLAGS to avoid shell quoting issues with spaces.
ENCODED=""
for flag in "${RUSTFLAGS_ARRAY[@]}"; do
    if [[ -n "${ENCODED}" ]]; then
        ENCODED+=$'\x1f'
    fi
    ENCODED+="${flag}"
done

# ---------------------------------------------------------------------------
# Set up library paths for the codegen backend
# ---------------------------------------------------------------------------
EXTRA_LD="${CUDA_PATH}/nvvm/lib64:${CUDA_PATH}/lib64"
CODEGEN_DIR="$(dirname "${CODEGEN_SO}")"
export LD_LIBRARY_PATH="${CODEGEN_DIR}:${EXTRA_LD}:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# Run cargo build
# ---------------------------------------------------------------------------
RELEASE_FLAG=""
if [[ "${OPT_LEVEL}" != "0" ]]; then
    RELEASE_FLAG="--release"
fi

BUILD_OUTPUT="$(
    cd "${CRATE_DIR}"
    CARGO_ENCODED_RUSTFLAGS="${ENCODED}" \
    CARGO_FEATURE_NO_F16_F128=1 \
    cargo build \
        --lib \
        --message-format=json-render-diagnostics \
        -Zbuild-std=core,alloc \
        --target=nvptx64-nvidia-cuda \
        ${RELEASE_FLAG} \
        2>"${WORK_DIR}/stderr.log" || true
)"

BUILD_EXIT=$?
STDERR_LOG="${WORK_DIR}/stderr.log"

# ---------------------------------------------------------------------------
# Extract the artifact path from Cargo's JSON output
# ---------------------------------------------------------------------------
PTX_PATH=""
if [[ -n "${BUILD_OUTPUT}" ]]; then
    PTX_PATH="$(
        echo "${BUILD_OUTPUT}" \
        | grep '"reason":"compiler-artifact"' \
        | tail -1 \
        | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    if obj.get('reason') == 'compiler-artifact':
        for f in obj.get('filenames', []):
            if f.endswith('.ptx'):
                print(f)
                sys.exit(0)
" 2>/dev/null || true
    )"
fi

# For LLVM IR mode, look for .ll files instead.
if [[ "${EMIT}" == "llvm-ir" && -z "${PTX_PATH}" ]]; then
    PTX_PATH="$(find "${CRATE_DIR}" -name '*.ll' -path '*/nvptx64-nvidia-cuda/*' 2>/dev/null | head -1 || true)"
fi

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
if [[ -n "${PTX_PATH}" && -f "${PTX_PATH}" ]]; then
    cat "${PTX_PATH}"
else
    # Build failed, relay stderr so Godbolt shows the diagnostics.
    if [[ -f "${STDERR_LOG}" ]]; then
        cat "${STDERR_LOG}" >&2
    fi
    # Also dump any non-JSON lines from stdout (rustc sometimes puts
    # diagnostics there).
    if [[ -n "${BUILD_OUTPUT}" ]]; then
        echo "${BUILD_OUTPUT}" | grep -v '^\s*{' >&2 || true
    fi
    echo "error: compilation failed, no PTX output produced" >&2
    exit 1
fi
