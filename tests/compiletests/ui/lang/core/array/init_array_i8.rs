// Test creating an array.
// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_init_array_i8(o: *mut i8) {
    let array = [0i8; 4];
    unsafe {
        *o = array[1];
    }
}
