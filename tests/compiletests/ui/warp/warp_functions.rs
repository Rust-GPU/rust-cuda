// Test CUDA warp-level functions compile correctly
// build-pass

use cuda_std::kernel;
use cuda_std::warp;

#[kernel]
pub unsafe fn test_warp_functions() {
    unsafe {
        // Test lane ID function
        let _lane = warp::lane_id();

        // Test active mask function
        let _mask = warp::activemask();

        // Test warp sync with full mask
        warp::sync_warp(0xFFFFFFFF);

        // Test warp sync with partial mask
        warp::sync_warp(0x0000FFFF);
    }
}
