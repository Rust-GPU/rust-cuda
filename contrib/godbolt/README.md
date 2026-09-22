# Compiler Explorer (Godbolt) Integration for rust-cuda

This directory contains everything needed to add rust-cuda as a compiler on
[Compiler Explorer](https://compiler-explorer.com/) so that users can type
Rust GPU kernel code and see the resulting PTX assembly.

## How it works

Compiler Explorer expects a single "compiler" binary that reads source on
stdin or from a file and writes assembly to stdout.  Since rust-cuda has no
standalone compiler (the pipeline is `rustc` with a custom codegen backend
plus `cargo` for dependency resolution), the integration uses a small Rust
wrapper binary that:

1. Accepts a `.rs` file containing `#[kernel]` functions.
2. Creates a temporary Cargo project that depends on `cuda_std`.
3. Sets `CARGO_ENCODED_RUSTFLAGS` with the same flags `cuda_builder` uses
   (codegen backend path, `no_std` injection, `nvptx64-nvidia-cuda` target,
   `-Zbuild-std=core,alloc`, etc.).
4. Runs `cargo build` and parses the JSON output to locate the `.ptx` artifact.
5. Prints the PTX to stdout (or LLVM IR if `--emit=llvm-ir` is passed).
6. Forwards compiler diagnostics to stderr so CE displays them.

## Files

| File | Purpose |
|------|---------|
| `rust-cuda-wrapper/` | Rust crate for the wrapper binary CE invokes as the "compiler" |
| `rust-cuda.defaults.properties` | CE configuration (compiler type, flags, defaults) |
| `rust-cuda.amazon.properties` | CE instance-specific overrides for the AWS fleet |
| `install.sh` | Installs the pinned nightly, builds the codegen backend and the wrapper, and lays out the prefix |
| `test-kernel.rs` | Sample kernel with shared memory and thread indexing |

## Supported flags

| Flag | Description |
|------|-------------|
| `--emit=ptx` | Output PTX assembly (default) |
| `--emit=llvm-ir` | Output LLVM IR before libnvvm conversion |
| `--opt-level=0` | Disable optimisations |
| `--opt-level=3` | Enable optimisations (default) |
| `--gpu-arch=sm_XX` | Target GPU compute capability (default `sm_75` / Turing) |
| `--version` | Print version info |

## Testing locally

### Prerequisites

- CUDA toolkit installed (need `libnvvm` in `$CUDA_PATH/nvvm/lib64/`)
- The Rust nightly pinned in `rust-toolchain.toml` (`nightly-2026-04-02`)
- A built `librustc_codegen_nvvm.so`

### Quick test

```bash
# From the rust-cuda repo root, after building the codegen backend:
export RUST_CUDA_ROOT=/opt/compiler-explorer/rust-cuda   # or your install prefix
export CUDA_PATH=/usr/local/cuda

# Run install.sh first (or manually arrange the prefix):
./contrib/godbolt/install.sh

# Then test:
$RUST_CUDA_ROOT/bin/rust-cuda-wrapper contrib/godbolt/test-kernel.rs
```

You should see PTX assembly printed to stdout.

### Without install.sh

If you already have the codegen backend built in the workspace, you can
point the wrapper at the repo tree directly:

```bash
export RUST_CUDA_ROOT=/path/to/rust-cuda
# Ensure $RUST_CUDA_ROOT/lib/librustc_codegen_nvvm.so exists.

cd contrib/godbolt/rust-cuda-wrapper
cargo run --release -- ../test-kernel.rs
```

### Running the integration test

The wrapper crate ships a smoke test that compiles `test-kernel.rs`
end-to-end and asserts the output looks like PTX:

```bash
cd contrib/godbolt/rust-cuda-wrapper
cargo test                                       # skips without RUST_CUDA_ROOT
RUST_CUDA_ROOT=/path/to/rust-cuda cargo test     # runs the real build
```

## Submitting to Compiler Explorer

1. Open an issue on [compiler-explorer/compiler-explorer](https://github.com/compiler-explorer/compiler-explorer)
   proposing the new compiler, linking to this directory.
2. Open a PR on [compiler-explorer/infra](https://github.com/compiler-explorer/infra)
   that adds `install.sh` to the builder configuration.
3. Copy `rust-cuda.defaults.properties` into
   `etc/config/` in the compiler-explorer repo.
4. Copy `rust-cuda.amazon.properties` into the appropriate instance
   config directory.

Key things CE maintainers will want to verify:

- The wrapper is sandboxed (it only writes to `$TMPDIR` and cleans up).
- Build times are acceptable (first build is slow due to `-Zbuild-std`; subsequent
  builds reuse the sysroot cache).
- The CUDA toolkit / `libnvvm` licence permits redistribution on CE's
  infrastructure (NVIDIA's EULA generally allows this for development tools).

## Compilation pipeline

For reference, the full pipeline that the wrapper reproduces:

```
  User's .rs file
       |
       v
  [cargo build]
       |  --target=nvptx64-nvidia-cuda
       |  -Zbuild-std=core,alloc
       |  CARGO_ENCODED_RUSTFLAGS with -Zcodegen-backend=...
       v
  [rustc + rustc_codegen_nvvm]
       |  Compiles Rust -> NVVM IR (LLVM 7 bitcode dialect)
       v
  [libnvvm]  (from CUDA toolkit)
       |  Optimises NVVM IR -> PTX
       v
  .ptx file  (stdout)
```
