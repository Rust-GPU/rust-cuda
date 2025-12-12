use cust::device::Device;
use cust::event::{Event, EventFlags};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{AsyncCopyDestination, DeviceBuffer, LockedBuffer};
use cust::module::Module;
use cust::prelude::EventStatus;
use cust::stream::{Stream, StreamFlags};
use std::time::Instant;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn correct_output(data: &[u32], x: u32) -> bool {
    let not_matching_element = data.iter().enumerate().find(|&(_, &elem)| elem != x);

    match not_matching_element {
        Some((index, elem)) => println!("Error! data[{index}] = {elem}, ref = {x}"),
        None => println!("All elements of the array match the value!"),
    }

    not_matching_element.is_none()
}

fn main() -> Result<(), cust::error::CudaError> {
    // Set up the context, load the module, and create a stream to run kernels in.
    let _ctx = cust::quick_init();
    let device = Device::get_device(0).expect("Couldn't find Cuda supported devices!");
    println!("Device Name: {}", device.name().unwrap());

    let module = Module::from_ptx(PTX, &[]).expect("Module couldn't be init!");
    let increment = module
        .get_function("increment")
        .expect("Kernel function not found!");
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).expect("Stream couldn't be init!");

    const N: usize = 16 * 1024 * 1024;
    let value = 26;

    let blocks = BlockSize::xy(512, 1);
    let grids = GridSize::xy(N / blocks.x, 1);

    let start_event = Event::new(EventFlags::DEFAULT)?;
    let stop_event = Event::new(EventFlags::DEFAULT)?;

    // Create buffers for data on host-side
    // Ideally should be page-locked for efficiency
    let mut host_a = LockedBuffer::new(&0u32, N).expect("host array couldn't be initialized!");
    let mut device_a =
        DeviceBuffer::from_slice(&[u32::MAX; N]).expect("device array couldn't be initialized!");

    start_event
        .record(&stream)
        .expect("Failed to record start_event in the CUDA stream!");
    let start = Instant::now();

    // # Safety
    //
    // Until the stop_event is triggered:
    // 1. `host_a` is not being modified
    // 2. Both `device_a` and `host_a` are not deallocated
    // 3. Until `stop_query` yields `EventStatus::Ready`, `device_a` is not involved in any other operation
    //    other than those of the operations in the stream.
    unsafe {
        device_a
            .async_copy_from(&host_a, &stream)
            .expect("Could not copy from host to device!");
    }

    // # Safety
    //
    // Number of threads * number of blocks = total number of elements.
    // Hence there will not be any out-of-bounds issues.
    unsafe {
        let result = launch!(increment<<<grids, blocks, 0, stream>>>(
            device_a.as_device_ptr(),
            value
        ));
        result.expect("Result of `increment` kernel did not process!");
    }

    // # Safety
    //
    // Until the stop_event is triggered:
    // 1. `device_a` is not being modified
    // 2. Both `device_a` and `host_a` are not deallocated
    // 3. At this point, until `stop_query` yields `EventStatus::Ready`,
    //    `host_a` is not involved in any other operation.
    unsafe {
        device_a
            .async_copy_to(&mut host_a, &stream)
            .expect("Could not copy from device to host!");
    }

    stop_event
        .record(&stream)
        .expect("Failed to record stop_event in the CUDA stream!");
    let cpu_time: u128 = start.elapsed().as_micros();

    let mut counter: u64 = 0;
    while stop_event.query() != Ok(EventStatus::Ready) {
        counter += 1
    }

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

    println!("test PASSED");
    Ok(())

    // The events and the memory buffers are automatically dropped here.
}
