use std::process::Command;

#[test]
fn compiles_test_kernel_to_ptx() {
    if std::env::var("RUST_CUDA_ROOT").is_err() {
        eprintln!("skipping: RUST_CUDA_ROOT not set");
        return;
    }

    let bin = env!("CARGO_BIN_EXE_rust-cuda-wrapper");
    let kernel = concat!(env!("CARGO_MANIFEST_DIR"), "/../test-kernel.rs");

    let output = Command::new(bin)
        .arg(kernel)
        .output()
        .expect("failed to run wrapper");

    assert!(
        output.status.success(),
        "wrapper exited non-zero. stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(".version"),
        "stdout doesn't look like PTX. got:\n{}",
        stdout
    );
}
