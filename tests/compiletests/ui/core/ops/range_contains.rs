// build-pass

use cuda_std::kernel;

fn has_two_decimal_digits(x: u32) -> bool {
    (10..100).contains(&x)
}

#[kernel]
pub unsafe fn main(i: u32, o: *mut u32) {
    unsafe {
        *o = has_two_decimal_digits(i) as u32;
    }
}
