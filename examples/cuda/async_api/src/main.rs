
use cust::memory::{DeviceBuffer, LockedBuffer, AsyncCopyDestination};
use cust::event::{Event, EventFlags};
use cust::prelude::EventStatus;
use cust::stream::{Stream, StreamFlags};
use cust::module::Module;
use cust::context::Context;
use cust::{launch, CudaFlags};
use cust::device::Device;
use cust::function::{GridSize, BlockSize};
use std::time::Instant;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn correct_output(data: &[u32], x: u32) -> bool {
    let not_matching_element = data
        .iter()
        .enumerate()
        .find(|&(_, &elem)| elem != x);

    match not_matching_element {
        Some((index, elem)) => println!("Error! data[{index}] = {elem}, ref = {x}"),
        None => println!("All elements of the array match the value!")
    }

    not_matching_element.is_none()
}

fn main() -> Result<(), cust::error::CudaError> {
    cust::init(CudaFlags::empty()).expect("Couldn't initialize CUDA environment!");

    let device = Device::get_device(0)
        .expect("Couldn't find Cuda supported devices!");

    println!("Device Name: {}", device.name().unwrap());

    // Set up the context, load the module, and create a stream to run kernels in.
    let _ctx = Context::new(device);
    let module = Module::from_ptx(PTX, &[]).expect("Module couldn't be init!");
    let increment = module.get_function("increment").expect("Kernel function not found!");
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Stream couldn't be init!");

    const N: usize = 16 * 1024 * 1024;
    const N_BYTES: usize = N * (i32::BITS as usize);
    let value = 26;

    let blocks = BlockSize::xy(512, 1);
    let grids = GridSize::xy((N / (blocks.x as usize)).try_into().unwrap(), 1);

    let start_event = Event::new(EventFlags::DEFAULT)?;
    let stop_event = Event::new(EventFlags::DEFAULT)?;

    // Create buffers for data on host-side
    // Ideally must be page-locked for efficiency
    let mut host_a = LockedBuffer::new(&0u32, N).expect("host array couldn't be initialized!");
    let mut device_a = DeviceBuffer::from_slice(&[u32::MAX; N]).expect("device array couldn't be initialized!");

    start_event.record(&stream).expect("Failed to record start_event in the CUDA stream!");
    let start = Instant::now();

    // SAFETY: until the stop_event being triggered:
    // 1. `host_a` is not being modified
    // 2. Both `device_a` and `host_a` are not deallocated
    // 3. Until `stop_query` yields `EventStatus::Ready`, `device_a` is not involved in any other operation
    //    other than those of the operations in the stream.
    unsafe {
        device_a.async_copy_from(&host_a, &stream).expect("Could not copy from host to device!");
    }

    // SAFETY: number of threads * number of blocks = total number of elements.
    // Hence there will not be any out-of-bounds issues.
    unsafe {
        let result = launch!(increment<<<grids, blocks, 0, stream>>>(
            device_a.as_device_ptr(),
            value
        ));
        result.expect("Result of `increment` kernel did not process!");
    }

    // SAFETY: until the stop_event being triggered:
    // 1. `device_a` is not being modified
    // 2. Both `device_a` and `host_a` are not deallocated
    // 3. At this point, until `stop_query` yields `EventStatus::Ready`, 
    //    `host_a` is not involved in any other operation.
    unsafe {
        device_a.async_copy_to(&mut host_a, &stream).expect("Could not copy from device to host!");
    }

    stop_event.record(&stream).expect("Failed to record stop_event in the CUDA stream!");
    let cpu_time: u128 = start.elapsed().as_micros();

    let mut counter: u64 = 0;
    while stop_event.query() != Ok(EventStatus::Ready) { counter += 1 }

    let gpu_time: u128 = stop_event
        .elapsed(&start_event)
        .expect("Failed to calculate duration of GPU operations!")
        .as_micros();
    
    println!("Time spent executing by the GPU: {gpu_time} microseconds");
    println!("Time spent by CPU in CUDA calls: {cpu_time} microseconds");
    println!("CPU executed {counter} iterations while waiting for GPU to finish.");
    
    assert!(correct_output(host_a.as_slice(), value));

    // Stream is synchronized as a safety measure
    stream.synchronize().expect("Stream couldn't synchronize!");

    // Events and buffers can be safely dropped now
    match Event::drop(start_event) {
        Ok(()) => println!("Successfully destroyed start_event"),
        Err((cuda_error, _event)) => {
            println!("Failed to destroy start_event: {:?}", cuda_error);
        },
    }

    match Event::drop(stop_event) {
        Ok(()) => println!("Successfully destroyed stop_event"),
        Err((cuda_error, _event)) => {
            println!("Failed to destroy stop_event: {:?}", cuda_error);
        },
    }

    DeviceBuffer::drop(device_a).expect("Couldn't drop device array!");
    LockedBuffer::drop(host_a).expect("Couldn't drop host array!");

    println!("test PASSED");
    Ok(())
}
