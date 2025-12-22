use std::error::Error;
use std::fmt;
use std::time::Duration;

use rand::Rng;

use blastoff::CublasContext;
use cust::event;
use cust::function;
use cust::launch;
use cust::memory::{self, CopyDestination as _};
use cust::module::Module;
use cust::stream;
use cust::util::SliceExt as _;

const VECTORS_LEN: usize = 10_000_000;
const NUM_WARMUP: usize = 100;
const NUM_RUNS: usize = 1000;
const BLOCK_SIZE: u32 = 1024;
const GRID_SIZE: u32 = 80;

static PTX_NATIVE: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels_cuda.ptx"));
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

struct KernelLaunchStats {
    /// The size of the block grid
    pub grid_size: u32,
    /// Number of threads per block
    pub block_size: u32,
    /// The amount of dynamically allocated shared memory
    pub shared_mem_size: u32,
    /// Number of registers used
    pub num_regs: u32,
}

impl fmt::Display for KernelLaunchStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Kernel launch stats:")?;
        writeln!(f, "    grid_size: {}", self.grid_size)?;
        writeln!(f, "    block_size: {}", self.block_size)?;
        writeln!(f, "    shared_mem_size: {}", self.shared_mem_size)?;
        writeln!(f, "    num_regs: {}", self.num_regs)
    }
}

/// Launch statistics and outputs
struct RunResult {
    /// The average value of the result
    pub res_average: f64,
    /// Duration of one iteration
    pub run_duration: Duration,
    /// Statistics of the running kernel
    pub kernel_launch_stats: Option<KernelLaunchStats>,
}

impl fmt::Display for RunResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "result: {}", self.res_average)?;
        writeln!(
            f,
            "Duration of one iteration: {} ms",
            self.run_duration.as_secs_f64() * 1000f64
        )?;
        if let Some(kernel_lunch_stats) = &self.kernel_launch_stats {
            write!(f, "{}", kernel_lunch_stats)?;
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize CUDA
    let _ctx = cust::quick_init()?;

    // make a CUDA stream to issue calls to.
    let stream = stream::Stream::new(stream::StreamFlags::NON_BLOCKING, None)?;

    // Generate input hosts vectors A and B
    let mut rng = rand::rng();
    let x_host = (0..VECTORS_LEN)
        .map(|_i| rng.random_range(0.0..=1.0f32))
        .collect::<Vec<_>>();
    let y_host = (0..VECTORS_LEN)
        .map(|_i| rng.random_range(0.0..=1.0f32))
        .collect::<Vec<_>>();

    // Allocate the GPU memory
    let x_gpu = x_host.as_slice().as_dbuf()?;
    let y_gpu = y_host.as_slice().as_dbuf()?;

    // Run cuBLAS test
    let blas_res = run_cublas_sdot_test(&stream, &x_gpu, &y_gpu)?;
    println!("cuBLAS:\n{}", blas_res);

    // Run native CUDA kernel test
    let module_native = Module::from_ptx(PTX_NATIVE, &[])?;
    let sdot_native = module_native.get_function("sdot")?;

    let native_res =
        run_cuda_sdot_test(&stream, sdot_native, &x_gpu, &y_gpu, GRID_SIZE, BLOCK_SIZE)?;
    println!("Native CUDA:\n{}", native_res);

    // Run Rust CUDA kernel test
    let module_rust = Module::from_ptx(PTX, &[])?;
    let sdot_rust = module_rust.get_function("sdot")?;

    let rust_res = run_cuda_sdot_test(&stream, sdot_rust, &x_gpu, &y_gpu, GRID_SIZE, BLOCK_SIZE)?;
    println!("Rust CUDA:\n{}", rust_res);

    Ok(())
}

/// Runs cuBLAS dot product of two vectors
///
/// Runs the scalar product calculations several times, before warming up
/// and calculating the execution time.
fn run_cublas_sdot_test(
    stream: &stream::Stream,
    x: &memory::DeviceBuffer<f32>,
    y: &memory::DeviceBuffer<f32>,
) -> Result<RunResult, Box<dyn Error>> {
    let mut ctx = CublasContext::new()?;

    // WarmUp
    for _ in 0..NUM_WARMUP {
        let mut result = memory::DeviceBox::new(&0.0)?;
        ctx.dot(stream, x.len(), x, y, &mut result)?;
        stream.synchronize()?;
        let _res = result.as_host_value()?;
    }

    // Run bench
    let mut res_average = 0f64;
    let begin = event::Event::new(event::EventFlags::DEFAULT)?;
    let end = event::Event::new(event::EventFlags::DEFAULT)?;
    begin.record(stream)?;

    for _ in 0..NUM_RUNS {
        let mut result = memory::DeviceBox::new(&0.0)?;
        ctx.dot(stream, x.len(), x, y, &mut result)?;
        stream.synchronize()?;
        res_average += result.as_host_value()? as f64;
    }

    end.record(stream)?;
    begin.synchronize()?;
    end.synchronize()?;

    res_average /= NUM_RUNS as f64;
    let run_duration = end.elapsed(&begin)?.div_f64(NUM_RUNS as f64);

    let stats = RunResult {
        res_average,
        run_duration,
        kernel_launch_stats: None,
    };
    Ok(stats)
}

/// Runs CUDA kernel test: dot product of two vectors
///
/// Runs the scalar product calculations several times, before warming up
/// and calculating the execution time.
fn run_cuda_sdot_test(
    stream: &stream::Stream,
    sdot_fun: function::Function,
    x: &memory::DeviceBuffer<f32>,
    y: &memory::DeviceBuffer<f32>,
    grid_size: u32,
    block_size: u32,
) -> Result<RunResult, Box<dyn Error>> {
    // Allocate memory to collect results, one per thread block
    let mut out_host = vec![0.0f32; grid_size as _];
    let out_gpu = memory::DeviceBuffer::zeroed(grid_size as _)?;

    // Shared memory size per thread block
    let shared_mem_size = block_size * (std::mem::size_of::<f32>() as u32);

    // WarmUp
    for _ in 0..NUM_WARMUP {
        unsafe {
            launch!(
                sdot_fun<<<grid_size, block_size, shared_mem_size, stream>>>(
                    x.as_device_ptr(),
                    x.len(),
                    y.as_device_ptr(),
                    y.len(),
                    out_gpu.as_device_ptr(),
                )
            )?;
        }
        stream.synchronize()?;
        out_gpu.copy_to(&mut out_host)?;
        let _res: f64 = out_host.iter().map(|e| *e as f64).sum();
    }

    // Run bench
    let mut res_average = 0f64;
    let begin = event::Event::new(event::EventFlags::DEFAULT)?;
    let end = event::Event::new(event::EventFlags::DEFAULT)?;
    begin.record(stream)?;

    for _ in 0..NUM_RUNS {
        unsafe {
            launch!(
                sdot_fun<<<grid_size, block_size, shared_mem_size, stream>>>(
                    x.as_device_ptr(),
                    x.len(),
                    y.as_device_ptr(),
                    y.len(),
                    out_gpu.as_device_ptr(),
                )
            )?;
        }
        stream.synchronize()?;
        out_gpu.copy_to(&mut out_host)?;
        let res: f64 = out_host.iter().map(|e| *e as f64).sum();
        res_average += res;
    }

    let kernel_launch_stats = KernelLaunchStats {
        grid_size,
        block_size,
        shared_mem_size,
        num_regs: sdot_fun.get_attribute(function::FunctionAttribute::NumRegisters)? as u32,
    };

    end.record(stream)?;
    begin.synchronize()?;
    end.synchronize()?;

    res_average /= NUM_RUNS as f64;
    let run_duration = end.elapsed(&begin)?.div_f64(NUM_RUNS as f64);

    let stats = RunResult {
        res_average,
        run_duration,
        kernel_launch_stats: Some(kernel_launch_stats),
    };

    Ok(stats)
}
