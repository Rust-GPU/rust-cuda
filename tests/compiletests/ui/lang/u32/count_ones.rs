// Test all trailing and leading zeros. No need to test ones, they just call the zero variant with !value

// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn count_ones_u8(buffer: *const u8, out: *mut u32) {
    unsafe {
        *out = (*buffer).count_ones();
    }
}

#[kernel]
pub unsafe fn count_ones_u16(buffer: *const u16, out: *mut u32) {
    unsafe {
        *out = (*buffer).count_ones();
    }
}

#[kernel]
pub unsafe fn count_ones_u32(buffer: *const u32, out: *mut u32) {
    unsafe {
        *out = (*buffer).count_ones();
    }
}

#[kernel]
pub unsafe fn count_ones_u64(buffer: *const u64, out: *mut u32) {
    unsafe {
        *out = (*buffer).count_ones();
    }
}

#[kernel]
pub unsafe fn count_ones_i8(buffer: *const i8, out: *mut u32) {
    unsafe {
        *out = (*buffer).count_ones();
    }
}

#[kernel]
pub unsafe fn count_ones_i16(buffer: *const i16, out: *mut u32) {
    unsafe {
        *out = (*buffer).count_ones();
    }
}

#[kernel]
pub unsafe fn count_ones_i32(buffer: *const i32, out: *mut u32) {
    unsafe {
        *out = (*buffer).count_ones();
    }
}

#[kernel]
pub unsafe fn count_ones_i64(buffer: *const i64, out: *mut u32) {
    unsafe {
        *out = (*buffer).count_ones();
    }
}
