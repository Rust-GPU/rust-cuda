#![allow(
    improper_ctypes_definitions,
    clippy::missing_safety_doc,
    unsafe_op_in_unsafe_fn
)]

use cuda_std::atomic::{intrinsics, mid};
use cuda_std::prelude::*;

#[kernel]
pub unsafe fn test_kernel(data: *mut i32) {
    let tid = (thread::block_dim_x() * thread::block_idx_x() + thread::thread_idx_x()) as i32;

    // Arithmetic atomics

    intrinsics::atomic_fetch_add_relaxed_i32_device(data.add(0), 10);
    intrinsics::atomic_fetch_sub_relaxed_i32_device(data.add(1), 10);
    intrinsics::atomic_fetch_exch_relaxed_i32_device(data.add(2), tid);
    intrinsics::atomic_fetch_max_relaxed_i32_device(data.add(3), tid);
    intrinsics::atomic_fetch_min_relaxed_i32_device(data.add(4), tid);

    mid::atomic_inc_bounded_relaxed_u32_device(data.add(5) as *mut u32, 17);

    mid::atomic_dec_bounded_relaxed_u32_device(data.add(6) as *mut u32, 137);

    intrinsics::atomic_fetch_cas_relaxed_i32_device(data.add(7), tid - 1, tid);

    // Bitwise atomics

    intrinsics::atomic_fetch_and_relaxed_i32_device(data.add(8), 2 * tid + 7);

    // Match CUDA's `1 << tid` wrapping behaviour for tid >= 32 (PTX shl.b32 masks
    // the shift count to 5 bits, same as Rust's wrapping_shl).
    intrinsics::atomic_fetch_or_relaxed_i32_device(data.add(9), 1i32.wrapping_shl(tid as u32));

    intrinsics::atomic_fetch_xor_relaxed_i32_device(data.add(10), tid);
}
