use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn i128_ops(
    a: &[u128],
    b: &[u128],
    add_out: *mut u128,
    sub_out: *mut u128,
    mul_out: *mut u128,
    and_out: *mut u128,
    xor_out: *mut u128,
    shl_out: *mut u128,
    lshr_out: *mut u128,
    ashr_out: *mut u128,
    udiv_out: *mut u128,
    sdiv_out: *mut u128,
    urem_out: *mut u128,
    srem_out: *mut u128,
) {
    let idx = thread::index_1d() as usize;
    if idx >= a.len() || idx >= b.len() {
        return;
    }

    let av = a[idx];
    let bv = b[idx];
    let shift = (bv & 127) as u32;
    let signed = av as i128;
    let signed_b = bv as i128;

    unsafe {
        *add_out.add(idx) = av.wrapping_add(bv);
        *sub_out.add(idx) = av.wrapping_sub(bv);
        *mul_out.add(idx) = av.wrapping_mul(bv);
        *and_out.add(idx) = av & bv;
        *xor_out.add(idx) = av ^ bv;
        *shl_out.add(idx) = av.wrapping_shl(shift);
        *lshr_out.add(idx) = av.wrapping_shr(shift);
        *ashr_out.add(idx) = (signed.wrapping_shr(shift)) as u128;
        *udiv_out.add(idx) = av / bv;
        *sdiv_out.add(idx) = (signed / signed_b) as u128;
        *urem_out.add(idx) = av % bv;
        *srem_out.add(idx) = (signed % signed_b) as u128;
    }
}
