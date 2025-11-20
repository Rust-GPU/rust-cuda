# Compiletests for Rust CUDA

This directory contains compile tests for the Rust CUDA project using the `compiletest` framework.

The code in these tests is not executed. Tests check that the compiler compiles
correctly. Tests in `dis/` verify correct PTX output.

## Running Tests

You can run the tests using the cargo alias:

```bash
cargo compiletest
```

Or run directly from this directory:

```bash
cargo run --release
```

### Options

- `--bless` - Update expected output files
- `--target-arch=compute_61,compute_75,compute_90` - Test multiple CUDA compute capabilities (comma-separated)
- Filter by test name: `cargo compiletest simple`
- `RUST_LOG=info` - Enable progress logging
- `RUST_LOG=debug` - Enable detailed debug logging

### Architecture-Specific Tests

Tests can target specific architectures using stage IDs:

```rust
// only-compute_70   - Only run on compute_70
// only-compute_90   - Only run on compute_90
// ignore-compute_61 - Skip on compute_61
```

## Multi-Architecture Testing

Test against multiple CUDA architectures:

```bash
cargo compiletest -- --target-arch=compute_61,compute_70,compute_90
```

Each test runs for all specified architectures.

## Debugging

- Use `RUST_LOG=debug` for detailed test execution
- Check generated PTX in `target/compiletest-results/`
- Filter specific tests: `cargo compiletest simple`
