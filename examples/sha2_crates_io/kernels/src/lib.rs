use cuda_std::prelude::*;
use sha2::{Digest, Sha256, Sha512};

// One-shot API for SHA256
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn sha256_oneshot(input: &[u8], output: *mut [u8; 32]) {
    let idx = thread::index_1d();

    if idx == 0 {
        let hash = Sha256::digest(input);

        unsafe {
            let output_slice = &mut *output;
            output_slice.copy_from_slice(&hash);
        }
    }
}

// Incremental API for SHA256
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn sha256_incremental(input1: &[u8], input2: &[u8], output: *mut [u8; 32]) {
    let idx = thread::index_1d();

    if idx == 0 {
        let mut hasher = Sha256::new();
        hasher.update(input1);
        hasher.update(input2);
        let hash = hasher.finalize();

        unsafe {
            let output_slice = &mut *output;
            output_slice.copy_from_slice(&hash);
        }
    }
}

// One-shot API for SHA512
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn sha512_oneshot(input: &[u8], output: *mut [u8; 64]) {
    let idx = thread::index_1d();

    if idx == 0 {
        let hash = Sha512::digest(input);

        unsafe {
            let output_slice = &mut *output;
            output_slice.copy_from_slice(&hash);
        }
    }
}

// Incremental API for SHA512
#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn sha512_incremental(input: &[u8], output: *mut [u8; 64]) {
    let idx = thread::index_1d();

    if idx == 0 {
        let mut hasher = Sha512::new();
        hasher.update(input);
        let hash = hasher.finalize();

        unsafe {
            let output_slice = &mut *output;
            output_slice.copy_from_slice(&hash);
        }
    }
}
