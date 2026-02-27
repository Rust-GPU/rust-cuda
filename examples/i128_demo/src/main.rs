use cust::prelude::*;
use std::error::Error;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

const INPUT_PAIRS: [(u128, u128); 23] = [
    // Basic non-zero divisor sanity check.
    (0, 3),
    // Simple add/sub with small + patterned mask divisor.
    (1, 0x1111_1111_1111_1111_1111_1111_1111_1111),
    // Max magnitude dividend exercising unsigned wraparound with patterned divisor.
    (u128::MAX, 0x2222_2222_2222_2222_2222_2222_2222_2222),
    // Near max positive signed plus odd offset to stress signed division.
    (
        (1u128 << 127) + 123_456_789,
        0x3333_3333_3333_3333_3333_3333_3333_3333,
    ),
    // Mixed hex patterns to ensure bitwise ops and shifts propagate carries between halves.
    (
        0x0123_4567_89ab_cdef_0123_4567_89ab_cdef,
        0x4444_4444_4444_4444_4444_4444_4444_4444,
    ),
    // Alternating pattern stressing xor/and/or combinations across words.
    (
        0xfedc_ba98_7654_3210_fedc_ba98_7654_3210,
        0x5555_5555_5555_5555_5555_5555_5555_5555,
    ),
    // Low-half mask to hit low→high shift carries.
    (
        0x0000_ffff_0000_ffff_0000_ffff_0000_ffff,
        0x6666_6666_6666_6666_6666_6666_6666_6666,
    ),
    // Random-looking pattern to detect misplaced limb ordering.
    (
        0xabcd_ef12_3456_789a_bcde_f012_3456_789a,
        0x7777_7777_7777_7777_7777_7777_7777_7777,
    ),
    // Pure power-of-two vs small divisor for shifts and div edge cases.
    (1u128 << 127, 5),
    // Signed overflow boundary vs max unsigned divisor.
    (i128::MAX as u128, u128::MAX),
    // Distinct power-of-two limbs to check cross-term multiplications.
    (1u128 << 64, (1u128 << 63) + 1),
    // Odd division with high-bit divisor to stress udiv/sdiv paths.
    (u128::MAX / 3, 0x8000_0000_0000_0000_0000_0000_0000_0001),
    // Near-overflow positive dividends paired with non power-of-two divisors.
    (0x7fff_ffff_ffff_ffff_0000_0000_0000_0001, u128::MAX / 5),
    // Signed negative boundary mixed with patterned divisor.
    (
        0x8000_0000_0000_0000_8000_0000_0000_0000,
        0xffff_ffff_0000_0000_ffff_ffff_0000_0001,
    ),
    // Arbitrary large magnitudes to sanity check arithmetic stability.
    (
        123_456_789_012_345_678_901_234_567_890u128,
        (1u128 << 127) - 3,
    ),
    // Values near u128::MAX to ensure carry/borrow propagation.
    (u128::MAX - 999_999_999, u128::MAX - 2),
    // Exercises the lower-half cross-carry path in the mul emulation.
    (
        0x0000_0000_0000_0000_ffff_ffff_ffff_ffff,
        0x0000_0000_0000_0000_ffff_ffff_ffff_ffff,
    ),
    // Check emulated mul path edgecase.
    (u128::MAX, u128::MAX),
    // Shift exactly 64 bits with positive divisor.
    (
        0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        (1u128 << 120) | 64,
    ),
    // Shift exactly 64 bits with negative divisor to stress signed paths.
    (
        0xfedc_ba98_7654_3210_0123_4567_89ab_cdef,
        (1u128 << 127) | 64,
    ),
    // Shift just below the limb boundary.
    (
        0xaaaa_aaaa_5555_5555_ffff_ffff_0000_0000,
        (1u128 << 96) | 63,
    ),
    // Shift just above the limb boundary.
    (
        0x0001_0203_0405_0607_0809_0a0b_0c0d_0e0f,
        (1u128 << 80) | 65,
    ),
    // Maximum masked shift amount with high-bit divisor.
    (
        0xffff_0000_0000_0000_ffff_0000_0000_0001,
        (1u128 << 127) | 127,
    ),
];

fn main() -> Result<(), Box<dyn Error>> {
    let _ctx = cust::quick_init()?;

    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let kernel = module.get_function("i128_ops")?;

    let (host_a, host_b): (Vec<u128>, Vec<u128>) = INPUT_PAIRS.iter().copied().unzip();

    let len = host_a.len();
    assert_eq!(len, host_b.len());

    let a_gpu = DeviceBuffer::from_slice(&host_a)?;
    let b_gpu = DeviceBuffer::from_slice(&host_b)?;

    let add_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let sub_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let mul_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let and_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let xor_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let shl_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let lshr_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let ashr_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let udiv_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let sdiv_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let urem_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;
    let srem_gpu = DeviceBuffer::from_slice(&vec![0u128; len])?;

    let block_size = 128usize;
    let grid_size = len.div_ceil(block_size);

    unsafe {
        launch!(
            kernel<<<grid_size, block_size, 0, stream>>>(
                a_gpu.as_device_ptr(),
                a_gpu.len(),
                b_gpu.as_device_ptr(),
                b_gpu.len(),
                add_gpu.as_device_ptr(),
                sub_gpu.as_device_ptr(),
                mul_gpu.as_device_ptr(),
                and_gpu.as_device_ptr(),
                xor_gpu.as_device_ptr(),
                shl_gpu.as_device_ptr(),
                lshr_gpu.as_device_ptr(),
                ashr_gpu.as_device_ptr(),
                udiv_gpu.as_device_ptr(),
                sdiv_gpu.as_device_ptr(),
                urem_gpu.as_device_ptr(),
                srem_gpu.as_device_ptr()
            )
        )?;
    }

    stream.synchronize()?;

    let mut gpu_add = vec![0u128; len];
    let mut gpu_sub = vec![0u128; len];
    let mut gpu_mul = vec![0u128; len];
    let mut gpu_and = vec![0u128; len];
    let mut gpu_xor = vec![0u128; len];
    let mut gpu_shl = vec![0u128; len];
    let mut gpu_lshr = vec![0u128; len];
    let mut gpu_ashr = vec![0u128; len];
    let mut gpu_udiv = vec![0u128; len];
    let mut gpu_sdiv = vec![0u128; len];
    let mut gpu_urem = vec![0u128; len];
    let mut gpu_srem = vec![0u128; len];

    add_gpu.copy_to(&mut gpu_add)?;
    sub_gpu.copy_to(&mut gpu_sub)?;
    mul_gpu.copy_to(&mut gpu_mul)?;
    and_gpu.copy_to(&mut gpu_and)?;
    xor_gpu.copy_to(&mut gpu_xor)?;
    shl_gpu.copy_to(&mut gpu_shl)?;
    lshr_gpu.copy_to(&mut gpu_lshr)?;
    ashr_gpu.copy_to(&mut gpu_ashr)?;
    udiv_gpu.copy_to(&mut gpu_udiv)?;
    sdiv_gpu.copy_to(&mut gpu_sdiv)?;
    urem_gpu.copy_to(&mut gpu_urem)?;
    srem_gpu.copy_to(&mut gpu_srem)?;

    let mut cpu_add = vec![0u128; len];
    let mut cpu_sub = vec![0u128; len];
    let mut cpu_mul = vec![0u128; len];
    let mut cpu_and = vec![0u128; len];
    let mut cpu_xor = vec![0u128; len];
    let mut cpu_shl = vec![0u128; len];
    let mut cpu_lshr = vec![0u128; len];
    let mut cpu_ashr = vec![0u128; len];
    let mut cpu_udiv = vec![0u128; len];
    let mut cpu_sdiv = vec![0u128; len];
    let mut cpu_urem = vec![0u128; len];
    let mut cpu_srem = vec![0u128; len];

    for (i, (&av, &bv)) in host_a.iter().zip(host_b.iter()).enumerate() {
        let shift = (bv & 127) as u32;
        let signed = av as i128;
        let signed_b = bv as i128;
        cpu_add[i] = av.wrapping_add(bv);
        cpu_sub[i] = av.wrapping_sub(bv);
        cpu_mul[i] = av.wrapping_mul(bv);
        cpu_and[i] = av & bv;
        cpu_xor[i] = av ^ bv;
        cpu_shl[i] = av.wrapping_shl(shift);
        cpu_lshr[i] = av.wrapping_shr(shift);
        cpu_ashr[i] = (signed.wrapping_shr(shift)) as u128;
        cpu_udiv[i] = av / bv;
        cpu_sdiv[i] = (signed / signed_b) as u128;
        cpu_urem[i] = av % bv;
        cpu_srem[i] = (signed % signed_b) as u128;
    }

    let mut all_ok = true;
    all_ok &= compare_results("add", &gpu_add, &cpu_add);
    all_ok &= compare_results("sub", &gpu_sub, &cpu_sub);
    all_ok &= compare_results("mul", &gpu_mul, &cpu_mul);
    all_ok &= compare_results("and", &gpu_and, &cpu_and);
    all_ok &= compare_results("xor", &gpu_xor, &cpu_xor);
    all_ok &= compare_results("shl", &gpu_shl, &cpu_shl);
    all_ok &= compare_results("lshr", &gpu_lshr, &cpu_lshr);
    all_ok &= compare_results("ashr", &gpu_ashr, &cpu_ashr);
    all_ok &= compare_results("udiv", &gpu_udiv, &cpu_udiv);
    all_ok &= compare_results("sdiv", &gpu_sdiv, &cpu_sdiv);
    all_ok &= compare_results("urem", &gpu_urem, &cpu_urem);
    all_ok &= compare_results("srem", &gpu_srem, &cpu_srem);

    if !all_ok {
        return Err("Mismatch between GPU and CPU i128 results".into());
    }

    // Ensure signed overflow (`i128::MIN / -1`) traps on the device.
    let trap_stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let trap_a = DeviceBuffer::from_slice(&[i128::MIN as u128])?;
    let trap_b = DeviceBuffer::from_slice(&[u128::MAX])?;
    let trap_add = DeviceBuffer::from_slice(&[0u128])?;
    let trap_sub = DeviceBuffer::from_slice(&[0u128])?;
    let trap_mul = DeviceBuffer::from_slice(&[0u128])?;
    let trap_and = DeviceBuffer::from_slice(&[0u128])?;
    let trap_xor = DeviceBuffer::from_slice(&[0u128])?;
    let trap_shl = DeviceBuffer::from_slice(&[0u128])?;
    let trap_lshr = DeviceBuffer::from_slice(&[0u128])?;
    let trap_ashr = DeviceBuffer::from_slice(&[0u128])?;
    let trap_udiv = DeviceBuffer::from_slice(&[0u128])?;
    let trap_sdiv = DeviceBuffer::from_slice(&[0u128])?;
    let trap_urem = DeviceBuffer::from_slice(&[0u128])?;
    let trap_srem = DeviceBuffer::from_slice(&[0u128])?;

    let trap_launch = unsafe {
        launch!(
            kernel<<<1, 1, 0, trap_stream>>>(
                trap_a.as_device_ptr(),
                trap_a.len(),
                trap_b.as_device_ptr(),
                trap_b.len(),
                trap_add.as_device_ptr(),
                trap_sub.as_device_ptr(),
                trap_mul.as_device_ptr(),
                trap_and.as_device_ptr(),
                trap_xor.as_device_ptr(),
                trap_shl.as_device_ptr(),
                trap_lshr.as_device_ptr(),
                trap_ashr.as_device_ptr(),
                trap_udiv.as_device_ptr(),
                trap_sdiv.as_device_ptr(),
                trap_urem.as_device_ptr(),
                trap_srem.as_device_ptr()
            )
        )
    };

    let trap_result = match trap_launch {
        Ok(()) => trap_stream.synchronize(),
        Err(e) => Err(e),
    };

    match trap_result {
        Err(e) => println!("Correctly got expected trap for i128::MIN / -1: {e}"),
        Ok(()) => return Err("Expected trap for i128::MIN / -1 not triggered".into()),
    }

    println!("All i128 GPU results match CPU computations.");
    Ok(())
}

fn compare_results(name: &str, gpu: &[u128], cpu: &[u128]) -> bool {
    let mut ok = true;
    for (idx, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        if g != c {
            println!("[{name}] mismatch at index {idx}: gpu={g:#034x}, cpu={c:#034x}");
            ok = false;
        }
    }

    if ok {
        println!("[{name}] results match");
    }

    ok
}
