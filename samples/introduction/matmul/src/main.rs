/* This example demonstrates an implementation of matrix multiplication.
 *
 * 1. The matrices are first created on the host side and then copied to the device.
 * 2. A shared piece of block-specific memory is created (on the device side), so that summation can be done very quickly
 * 3. The result is copied back to host, where the accumulated error occur.
 * 4. Extra: The error that accumulates during the summation process is reduced (in the kernel itself) using [Kahan summation algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm).
 */

use cuda_std::glam::USizeVec2;
use cust::device::Device;
use cust::event::{Event, EventFlags};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn matrix_multiply(
    block_size: usize,
    dims_a: USizeVec2,
    dims_b: USizeVec2,
) -> Result<(), cust::error::CudaError> {
    let dims_c = USizeVec2::new(dims_b.x, dims_a.y);
    let size_a = dims_a.x * dims_a.y;
    let h_a = LockedBuffer::new(&1.0f32, size_a).expect("host array couldn't be initialized!");

    let size_b = dims_b.x * dims_b.y;
    let h_b = LockedBuffer::new(&0.01f32, size_b).expect("host arrray couldn't be initialized!");

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Stream couldn't be init!");

    let size_c = dims_b.x * dims_a.y;
    let mut h_c = LockedBuffer::new(&0.0f32, size_c).expect("host array couldn't be initialized!");

    let start_event = Event::new(EventFlags::DEFAULT)?;
    let stop_event = Event::new(EventFlags::DEFAULT)?;

    let d_a =
        DeviceBuffer::from_slice(h_a.as_slice()).expect("device array couldn't be initialized!");
    let d_b =
        DeviceBuffer::from_slice(h_b.as_slice()).expect("device array couldn't be initialized!");
    let d_c =
        DeviceBuffer::from_slice(h_c.as_slice()).expect("device array couldn't be initialized!");

    stream.synchronize().expect("Stream couldn't synchronize!");
    let threads = BlockSize::xy(block_size as u32, block_size as u32);
    let grid = GridSize::xy(
        (dims_b.x / (threads.x as usize)).try_into().unwrap(),
        (dims_a.y / (threads.y as usize)).try_into().unwrap(),
    );

    println!("Computing result using CUDA Kernel...");

    let module = Module::from_ptx(PTX, &[]).expect("Module couldn't be init!");
    let matrix_mul_cuda = module
        .get_function("matrix_mul_cuda")
        .expect("Kernel function not found!");

    unsafe {
        // The function definition of the kernel is:
        // ```
        // pub unsafe fn matrix_mul_cuda(c: *mut f32, a: &[f32], b: &[f32], wa: usize, wb: usize)
        // ```
        // For elements that have the type `*mut T` or `*const T`, we'll need to pass only the device pointer.
        // For elements that have the type `&[T]`, we must pass the device pointer as well as the length of the slice.
        launch!(matrix_mul_cuda<<<grid, threads, 0, stream>>>(
            d_c.as_device_ptr(),
            d_a.as_device_ptr(),
            d_a.len(),
            d_b.as_device_ptr(),
            d_b.len(),
            dims_a.x,
            dims_b.x
        ))?;
    }

    println!("Done!");
    stream.synchronize().expect("Stream couldn't synchronize!");

    start_event
        .record(&stream)
        .expect("Failed to record start_event in the CUDA stream!");

    const N_ITER: u32 = 300;

    for _ in 0..N_ITER {
        unsafe {
            launch!(matrix_mul_cuda<<<grid, threads, 0, stream>>>(
                d_c.as_device_ptr(),
                d_a.as_device_ptr(),
                d_a.len(),
                d_b.as_device_ptr(),
                d_b.len(),
                dims_a.x,
                dims_b.x,
            ))?;
        }
    }

    stop_event
        .record(&stream)
        .expect("Failed to record stop_event in the CUDA stream!");

    stop_event
        .synchronize()
        .expect("Stream couldn't synchronize!");

    let gpu_time: u128 = stop_event
        .elapsed(&start_event)
        .expect("Failed to calculate duration of GPU operations!")
        .as_micros();

    let avg_time = gpu_time as f32 / N_ITER as f32;
    println!(
        "Average time spent executing by the GPU: {} microseconds",
        avg_time
    );
    let flops_per_matrix_mul = 2.0 * (dims_a.x as f32) * (dims_a.y as f32) * (dims_b.x as f32);
    let giga_flops = (flops_per_matrix_mul / (avg_time)) / 1000.0;
    println!("Performance = {} GFlop/s", giga_flops);

    unsafe {
        d_c.async_copy_to(&mut h_c, &stream)
            .expect("Could not copy from device to host!");
    }
    stream.synchronize().expect("Stream couldn't synchronize!");

    // checking computed result
    // test relative error by the formula
    // |<x, y>_cpu - <x, y>_gpu| / |<x, y>_cpu|
    let machine_epsilon = 1.1920929E-07f32;
    let mut correct = true;

    for i in 0..(dims_c.x * dims_c.y) {
        let abs_err = (h_c[i] - (dims_a.x as f32 * 0.01f32)).abs();
        let dot_length = (dims_a.x as f32).abs();
        let abs_val = h_c[i].abs();
        let rel_err = abs_err / abs_val.max(dot_length * machine_epsilon);

        if rel_err > 1e-6 {
            println!(
                "Error at index {}: CPU = {}, GPU = {}, rel_err = {}",
                i,
                dims_a.x as f32 * 0.01f32,
                h_c[i],
                rel_err
            );
            correct = false;
        }
    }

    if correct {
        println!("Result = PASS");
        println!(
            "NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled."
        );
    } else {
        println!("Result = FAIL");
        return Err(cust::error::CudaError::UnknownError);
    }

    Ok(())
}

fn main() -> Result<(), cust::error::CudaError> {
    // Set up the context, load the module, and create a stream to run kernels in.
    let _ctx = cust::quick_init();
    let device = Device::get_device(0).expect("Couldn't find Cuda supported devices!");
    println!("Device Name: {}", device.name().unwrap());

    let block_size: u32 = 32;
    let dims_a = USizeVec2::new(40 * block_size as usize, 40 * block_size as usize);
    let dims_b = USizeVec2::new(80 * block_size as usize, 40 * block_size as usize);

    if dims_a.x != dims_b.y {
        panic!("Matrix multiplication not possible with the given dimensions!");
    }

    matrix_multiply(block_size as usize, dims_a, dims_b)?;
    Ok(())
}
