// Test creating an array.
// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_init_array_i32(o: *mut i32) {
    let array = [0i32; 4];
    unsafe {
        *o = array[1];
    }
}
