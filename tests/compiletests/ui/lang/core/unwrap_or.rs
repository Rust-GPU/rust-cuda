#![crate_name = "unwrap_or"]

// unwrap_or generates some memory-bools (as u8). Test to make sure they're fused away.

// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_unwrap_or(out: *mut u32) {
    unsafe {
        *out = None.unwrap_or(15);
    }
}
