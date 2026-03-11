// Test creating an array.
// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn test_init_array_i16(o: *mut i16) {
    let array = [0i16; 4];
    unsafe {
        *o = array[1];
    }
}
