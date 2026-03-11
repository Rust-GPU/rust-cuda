// Test all trailing and leading zeros. No need to test ones, they just call the zero variant with !value

// build-pass

use cuda_std::kernel;

#[kernel]
pub unsafe fn reverse_bits_u8(buffer: *const u8, out: *mut u8) {
    unsafe {
        *out = (*buffer).reverse_bits();
    }
}

#[kernel]
pub unsafe fn reverse_bits_u16(buffer: *const u16, out: *mut u16) {
    unsafe {
        *out = (*buffer).reverse_bits();
    }
}

#[kernel]
pub unsafe fn reverse_bits_u32(buffer: *const u32, out: *mut u32) {
    unsafe {
        *out = (*buffer).reverse_bits();
    }
}

#[kernel]
pub unsafe fn reverse_bits_u64(buffer: *const u64, out: *mut u64) {
    unsafe {
        *out = (*buffer).reverse_bits();
    }
}

#[kernel]
pub unsafe fn reverse_bits_i8(buffer: *const i8, out: *mut i8) {
    unsafe {
        *out = (*buffer).reverse_bits();
    }
}

#[kernel]
pub unsafe fn reverse_bits_i16(buffer: *const i16, out: *mut i16) {
    unsafe {
        *out = (*buffer).reverse_bits();
    }
}

#[kernel]
pub unsafe fn reverse_bits_i32(buffer: *const i32, out: *mut i32) {
    unsafe {
        *out = (*buffer).reverse_bits();
    }
}

#[kernel]
pub unsafe fn reverse_bits_i64(buffer: *const i64, out: *mut i64) {
    unsafe {
        *out = (*buffer).reverse_bits();
    }
}
