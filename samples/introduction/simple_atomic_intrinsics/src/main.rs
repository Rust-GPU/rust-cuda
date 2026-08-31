/* Demonstrates trivial use of global-memory atomic device functions, mirroring
 * NVIDIA's simpleAtomicIntrinsics CUDA sample.
 *
 * A 64×256 grid (16 384 threads) each performs eleven atomic operations on a
 * shared 11-element i32 array and the host verifies the results.
 */

use cust::memory::{CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use std::error::Error;
use std::time::Instant;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

const NUM_BLOCKS: u32 = 64;
const NUM_THREADS: u32 = 256;
const NUM_DATA: usize = 11;

fn compute_gold(gpu_data: &[i32; NUM_DATA], total_threads: usize) -> bool {
    let len = total_threads;
    let mut ok = true;

    // slot 0 – atomicAdd(+10): sum of len additions of 10
    let expected = 10 * len as i32;
    if gpu_data[0] != expected {
        println!("atomicAdd failed: expected {expected}, got {}", gpu_data[0]);
        ok = false;
    }

    // slot 1 – atomicSub(-10)
    let expected = -(10 * len as i32);
    if gpu_data[1] != expected {
        println!("atomicSub failed: expected {expected}, got {}", gpu_data[1]);
        ok = false;
    }

    // slot 2 – atomicExch: final value must be a valid tid in [0, len)
    if !(0..len as i32).contains(&gpu_data[2]) {
        println!("atomicExch failed: got {}", gpu_data[2]);
        ok = false;
    }

    // slot 3 – atomicMax: sequential max of 0..len starting from -(1<<8)
    let expected = {
        let mut v = -(1i32 << 8);
        for i in 0..len {
            v = v.max(i as i32);
        }
        v
    };
    if gpu_data[3] != expected {
        println!("atomicMax failed: expected {expected}, got {}", gpu_data[3]);
        ok = false;
    }

    // slot 4 – atomicMin
    let expected = {
        let mut v = 1i32 << 8;
        for i in 0..len {
            v = v.min(i as i32);
        }
        v
    };
    if gpu_data[4] != expected {
        println!("atomicMin failed: expected {expected}, got {}", gpu_data[4]);
        ok = false;
    }

    // slot 5 – atomicInc(limit=17): each thread does bounded inc, final value in [0, 16]
    if !(0..=16).contains(&gpu_data[5]) {
        println!("atomicInc failed: expected [0, 16], got {}", gpu_data[5]);
        ok = false;
    }

    // slot 6 – atomicDec(limit=137): each thread does bounded dec, final value in [0, 137]
    if !(0..=137).contains(&gpu_data[6]) {
        println!("atomicDec failed: expected [0, 137], got {}", gpu_data[6]);
        ok = false;
    }

    // slot 7 – atomicCAS: final value must be a valid tid in [0, len)
    if !(0..len as i32).contains(&gpu_data[7]) {
        println!("atomicCAS failed: got {}", gpu_data[7]);
        ok = false;
    }

    // slot 8 – atomicAnd(2*tid+7) starting from 0xff
    let expected = {
        let mut v = 0xffi32;
        for i in 0..len {
            v &= 2 * i as i32 + 7;
        }
        v
    };
    if gpu_data[8] != expected {
        println!("atomicAnd failed: expected {expected}, got {}", gpu_data[8]);
        ok = false;
    }

    // slot 9 – atomicOr(1<<tid) starting from 0.
    // For tid ≥ 32 the PTX shl.b32 wraps (modulo 32), same as wrapping_shl.
    let expected = {
        let mut v = 0i32;
        for i in 0..len {
            v |= 1i32.wrapping_shl(i as u32);
        }
        v
    };
    if gpu_data[9] != expected {
        println!("atomicOr failed: expected {expected}, got {}", gpu_data[9]);
        ok = false;
    }

    // slot 10 – atomicXor(tid) starting from 0xff
    let expected = {
        let mut v = 0xffi32;
        for i in 0..len {
            v ^= i as i32;
        }
        v
    };
    if gpu_data[10] != expected {
        println!(
            "atomicXor failed: expected {expected}, got {}",
            gpu_data[10]
        );
        ok = false;
    }

    ok
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("simpleAtomicIntrinsics starting...");

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let mut h_data = [0i32; NUM_DATA];
    // AND and XOR tests start with 0xff in their slots
    h_data[8] = 0xff;
    h_data[10] = 0xff;

    let d_data = DeviceBuffer::from_slice(&h_data)?;

    let kernel = module.get_function("test_kernel")?;

    let start = Instant::now();

    unsafe {
        cust::launch!(
            kernel<<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(d_data.as_device_ptr())
        )?;
    }

    stream.synchronize()?;

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    println!("Processing time: {elapsed_ms:.3} ms");

    d_data.copy_to(&mut h_data)?;

    let total_threads = (NUM_BLOCKS * NUM_THREADS) as usize;
    let passed = compute_gold(&h_data, total_threads);

    println!(
        "simpleAtomicIntrinsics completed, returned {}",
        if passed { "OK" } else { "ERROR!" }
    );

    if !passed {
        std::process::exit(1);
    }

    Ok(())
}
