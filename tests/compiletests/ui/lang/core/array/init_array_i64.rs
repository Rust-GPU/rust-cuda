// Test creating an array.
// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_init_array_i64(o: *mut i64) {
    let array = [0i64; 4];
    unsafe {
        *o = array[1];
    }
}
