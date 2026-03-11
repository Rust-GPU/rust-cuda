// Test CUDA warp shuffle functions compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::warp;

#[kernel]
pub unsafe fn test_warp_shuffle_functions() {
    let mask = 0xFFFFFFFF_u32; // Full warp mask
    let width = 32_u32; // Full warp width

    // Test warp_shuffle_xor with various types
    unsafe {
        // 8-bit types
        let val_i8: i8 = 42;
        let (res_i8, pred_i8) = warp::warp_shuffle_xor(mask, val_i8, 1, width);

        let val_u8: u8 = 42;
        let (res_u8, pred_u8) = warp::warp_shuffle_xor(mask, val_u8, 1, width);

        // 16-bit types
        let val_i16: i16 = 1234;
        let (res_i16, pred_i16) = warp::warp_shuffle_xor(mask, val_i16, 2, width);

        let val_u16: u16 = 1234;
        let (res_u16, pred_u16) = warp::warp_shuffle_xor(mask, val_u16, 2, width);

        // 32-bit types
        let val_i32: i32 = 123456;
        let (res_i32, pred_i32) = warp::warp_shuffle_xor(mask, val_i32, 4, width);

        let val_u32: u32 = 123456;
        let (res_u32, pred_u32) = warp::warp_shuffle_xor(mask, val_u32, 4, width);

        let val_f32: f32 = 3.14159;
        let (res_f32, pred_f32) = warp::warp_shuffle_xor(mask, val_f32, 8, width);

        // 64-bit types
        let val_i64: i64 = 1234567890;
        let (res_i64, pred_i64) = warp::warp_shuffle_xor(mask, val_i64, 16, width);

        let val_u64: u64 = 1234567890;
        let (res_u64, pred_u64) = warp::warp_shuffle_xor(mask, val_u64, 16, width);

        let val_f64: f64 = 2.718281828;
        let (res_f64, pred_f64) = warp::warp_shuffle_xor(mask, val_f64, 16, width);

        // 128-bit types
        let val_i128: i128 = 12345678901234567890;
        let (res_i128, pred_i128) = warp::warp_shuffle_xor(mask, val_i128, 1, width);

        let val_u128: u128 = 12345678901234567890;
        let (res_u128, pred_u128) = warp::warp_shuffle_xor(mask, val_u128, 1, width);
    }

    // Test warp_shuffle_down with various types
    unsafe {
        let delta = 1_u32;

        let val_i32: i32 = 42;
        let (res_i32, pred_i32) = warp::warp_shuffle_down(mask, val_i32, delta, width);

        let val_u32: u32 = 42;
        let (res_u32, pred_u32) = warp::warp_shuffle_down(mask, val_u32, delta, width);

        let val_f32: f32 = 1.0;
        let (res_f32, pred_f32) = warp::warp_shuffle_down(mask, val_f32, delta, width);

        let val_i64: i64 = 100;
        let (res_i64, pred_i64) = warp::warp_shuffle_down(mask, val_i64, delta, width);

        let val_u64: u64 = 100;
        let (res_u64, pred_u64) = warp::warp_shuffle_down(mask, val_u64, delta, width);

        let val_f64: f64 = 1.0;
        let (res_f64, pred_f64) = warp::warp_shuffle_down(mask, val_f64, delta, width);
    }

    // Test warp_shuffle_up with various types
    unsafe {
        let delta = 1_u32;

        let val_i32: i32 = 42;
        let (res_i32, pred_i32) = warp::warp_shuffle_up(mask, val_i32, delta, width);

        let val_u32: u32 = 42;
        let (res_u32, pred_u32) = warp::warp_shuffle_up(mask, val_u32, delta, width);

        let val_f32: f32 = 1.0;
        let (res_f32, pred_f32) = warp::warp_shuffle_up(mask, val_f32, delta, width);

        let val_i64: i64 = 100;
        let (res_i64, pred_i64) = warp::warp_shuffle_up(mask, val_i64, delta, width);

        let val_u64: u64 = 100;
        let (res_u64, pred_u64) = warp::warp_shuffle_up(mask, val_u64, delta, width);

        let val_f64: f64 = 1.0;
        let (res_f64, pred_f64) = warp::warp_shuffle_up(mask, val_f64, delta, width);
    }

    // Test warp_shuffle_idx with various types
    unsafe {
        let idx = 5_u32;

        let val_i32: i32 = 42;
        let (res_i32, pred_i32) = warp::warp_shuffle_idx(mask, val_i32, idx, width);

        let val_u32: u32 = 42;
        let (res_u32, pred_u32) = warp::warp_shuffle_idx(mask, val_u32, idx, width);

        let val_f32: f32 = 1.0;
        let (res_f32, pred_f32) = warp::warp_shuffle_idx(mask, val_f32, idx, width);

        let val_i64: i64 = 100;
        let (res_i64, pred_i64) = warp::warp_shuffle_idx(mask, val_i64, idx, width);

        let val_u64: u64 = 100;
        let (res_u64, pred_u64) = warp::warp_shuffle_idx(mask, val_u64, idx, width);

        let val_f64: f64 = 1.0;
        let (res_f64, pred_f64) = warp::warp_shuffle_idx(mask, val_f64, idx, width);
    }

    // Test with different mask values
    unsafe {
        let partial_mask = 0x0000FFFF_u32; // Lower 16 lanes
        let val: i32 = 123;
        let (res, pred) = warp::warp_shuffle_xor(partial_mask, val, 1, width);
    }

    // Test with different width values (must be power of 2 and <= 32)
    unsafe {
        let val: i32 = 456;
        let lane_mask = 1_u32;

        // Width 16
        let (res16, pred16) = warp::warp_shuffle_xor(mask, val, lane_mask, 16);

        // Width 8
        let (res8, pred8) = warp::warp_shuffle_xor(mask, val, lane_mask, 8);

        // Width 4
        let (res4, pred4) = warp::warp_shuffle_xor(mask, val, lane_mask, 4);

        // Width 2
        let (res2, pred2) = warp::warp_shuffle_xor(mask, val, lane_mask, 2);
    }

    // Test with half-precision floating point types (if available)
    #[cfg(feature = "half")]
    unsafe {
        use half::{bf16, f16};

        let val_f16 = f16::from_f32(1.5);
        let (res_f16, pred_f16) = warp::warp_shuffle_xor(mask, val_f16, 1, width);

        let val_bf16 = bf16::from_f32(2.5);
        let (res_bf16, pred_bf16) = warp::warp_shuffle_xor(mask, val_bf16, 1, width);
    }
}

// Test edge cases and boundary conditions
#[kernel]
pub unsafe fn test_warp_shuffle_edge_cases() {
    let mask = 0xFFFFFFFF_u32;

    // Test with lane_mask = 0 (should shuffle with same lane)
    unsafe {
        let val: i32 = 999;
        let (res, pred) = warp::warp_shuffle_xor(mask, val, 0, 32);
    }

    // Test with maximum lane_mask
    unsafe {
        let val: i32 = 888;
        let (res, pred) = warp::warp_shuffle_xor(mask, val, 31, 32);
    }

    // Test shuffle_down with delta = 0
    unsafe {
        let val: i32 = 777;
        let (res, pred) = warp::warp_shuffle_down(mask, val, 0, 32);
    }

    // Test shuffle_up with delta = 0
    unsafe {
        let val: i32 = 666;
        let (res, pred) = warp::warp_shuffle_up(mask, val, 0, 32);
    }

    // Test shuffle_idx with idx = 0 and idx = 31
    unsafe {
        let val: i32 = 555;
        let (res0, pred0) = warp::warp_shuffle_idx(mask, val, 0, 32);
        let (res31, pred31) = warp::warp_shuffle_idx(mask, val, 31, 32);
    }
}

// Test that the functions work in practical scenarios
#[kernel]
pub unsafe fn test_warp_shuffle_practical() {
    let lane_id = warp::lane_id();
    let mask = 0xFFFFFFFF_u32;

    // Butterfly reduction pattern using XOR shuffle
    unsafe {
        let mut val = lane_id as i32;

        // Stage 1: XOR with distance 16
        let (shuffled, _) = warp::warp_shuffle_xor(mask, val, 16, 32);
        val += shuffled;

        // Stage 2: XOR with distance 8
        let (shuffled, _) = warp::warp_shuffle_xor(mask, val, 8, 32);
        val += shuffled;

        // Stage 3: XOR with distance 4
        let (shuffled, _) = warp::warp_shuffle_xor(mask, val, 4, 32);
        val += shuffled;

        // Stage 4: XOR with distance 2
        let (shuffled, _) = warp::warp_shuffle_xor(mask, val, 2, 32);
        val += shuffled;

        // Stage 5: XOR with distance 1
        let (shuffled, _) = warp::warp_shuffle_xor(mask, val, 1, 32);
        val += shuffled;
    }

    // Broadcast from lane 0 using shuffle_idx
    unsafe {
        let my_val = lane_id * 10;
        let (broadcast_val, is_valid) = warp::warp_shuffle_idx(mask, my_val, 0, 32);
    }

    // Shift pattern using shuffle_down
    unsafe {
        let my_val = lane_id as f32;
        let (shifted_val, is_valid) = warp::warp_shuffle_down(mask, my_val, 1, 32);
    }

    // Reverse shift using shuffle_up
    unsafe {
        let my_val = (31 - lane_id) as f32;
        let (shifted_val, is_valid) = warp::warp_shuffle_up(mask, my_val, 1, 32);
    }
}
