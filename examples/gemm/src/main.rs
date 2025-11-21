//! Example demonstrating GEMM (General Matrix Multiply) on CUDA using Rust CUDA.
//!
//! This example benchmarks naive and tiled GEMM kernels as well as cuBLAS for various matrix sizes.
//! It uses the `cust` crate for CUDA management and `ndarray` for host-side matrix operations.

use std::cell;
use std::error::Error;

use cust::event;
use cust::launch;
use cust::memory;
use cust::memory::CopyDestination as _;
use cust::module;
use cust::stream;
use cust::util::SliceExt as _;
use ndarray::Array;
use ndarray_rand::RandomExt as _;
use ndarray_rand::rand_distr::Uniform;

const EPS: f32 = 0.01;
const NUM_WARMUPS: usize = 2;
const NUM_RUNS: usize = 10;
const MAT_SIZES: [usize; 8] = [32, 64, 128, 256, 512, 1024, 2048, 4096];
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

type GemmFn = dyn Fn(
    &stream::Stream,
    &module::Module,
    &memory::DeviceBuffer<f32>,
    &memory::DeviceBuffer<f32>,
    &mut memory::DeviceBuffer<f32>,
    usize,
    usize,
    usize,
    f32,
    f32,
) -> Result<(), Box<dyn Error>>;

fn main() -> Result<(), Box<dyn Error>> {
    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = module::Module::from_ptx(PTX, &[])?;

    // Make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = stream::Stream::new(stream::StreamFlags::NON_BLOCKING, None)?;

    run_cublas(&stream)?;
    run_gemm_kernel(&stream, &module, &gemm_naive, "gemm_naive")?;
    run_gemm_kernel(&stream, &module, &gemm_tiled, "gemm_tiled")?;

    Ok(())
}

/// Runs the cuBLAS GEMM for a set of matrix sizes and checks correctness.
///
/// # Arguments
/// * `stream` - CUDA stream to use for kernel launches and memory operations.
///
/// This function benchmarks cuBLAS GEMM and checks the result for small matrices.
fn run_cublas(stream: &stream::Stream) -> Result<(), Box<dyn Error>> {
    // Make a cuBLAS context which manages the cuBLAS internal GPU memory allocations.
    let mut cublas_ctx = blastoff::CublasContext::new()?;

    // Sanity check.
    {
        let mat_a = ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mat_b = ndarray::arr2(&[[5.0, 6.0], [7.0, 8.0]]);
        let mat_c_expect = ndarray::arr2(&[[19.0, 22.0], [43.0, 50.0]]);
        let (alpha, beta) = (1.0, 0.0);

        let mat_a_gpu = mat_a.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mat_b_gpu = mat_b.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mut mat_c_gpu = unsafe { cust::memory::DeviceBuffer::uninitialized(2 * 2)? };
        let alpha_gpu = cust::memory::DeviceBox::new(&alpha)?;
        let beta_gpu = cust::memory::DeviceBox::new(&beta)?;

        // ndarray uses row-major order, but cuBLAS uses column-major order.
        // In such case, C=AxB is equivalent to C^T=B^TxA^T.
        cublas_ctx.gemm::<f32>(
            stream,
            2,
            2,
            2,
            &alpha_gpu,
            &mat_b_gpu,
            2,
            blastoff::MatrixOp::None,
            &beta_gpu,
            &mat_a_gpu,
            2,
            blastoff::MatrixOp::None,
            &mut mat_c_gpu,
            2,
        )?;
        stream.synchronize()?;

        let mut mat_c_actual = Array::<f32, _>::zeros((2, 2));
        mat_c_gpu.copy_to(&mut mat_c_actual.as_slice_mut().unwrap())?;
        assert!(mat_c_expect.relative_eq(&mat_c_actual, EPS, EPS));
    }

    for sz in MAT_SIZES.iter().cloned() {
        let mat_a = ndarray::Array2::<f32>::random((sz, sz), Uniform::new(-10., 10.));
        let mat_b = ndarray::Array2::<f32>::random((sz, sz), Uniform::new(-10., 10.));
        let mat_c = ndarray::Array2::<f32>::random((sz, sz), Uniform::new(-10., 10.));
        let (alpha, beta) = (1.0, 0.0);

        let mat_a_gpu = mat_a.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mat_b_gpu = mat_b.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mut mat_c_gpu = mat_c.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let alpha_gpu = cust::memory::DeviceBox::new(&alpha)?;
        let beta_gpu = cust::memory::DeviceBox::new(&beta)?;
        stream.synchronize()?;

        // Warm up before timing.
        for _ in 0..NUM_WARMUPS {
            cublas_ctx.gemm::<f32>(
                stream,
                sz,
                sz,
                sz,
                &alpha_gpu,
                &mat_b_gpu,
                sz,
                blastoff::MatrixOp::None,
                &beta_gpu,
                &mat_a_gpu,
                sz,
                blastoff::MatrixOp::None,
                &mut mat_c_gpu,
                sz,
            )?;
        }
        stream.synchronize()?;

        // Time the kernel execution.
        let beg = event::Event::new(event::EventFlags::DEFAULT)?;
        let end = event::Event::new(event::EventFlags::DEFAULT)?;
        beg.record(stream)?;
        for _ in 0..NUM_RUNS {
            cublas_ctx.gemm::<f32>(
                stream,
                sz,
                sz,
                sz,
                &alpha_gpu,
                &mat_b_gpu,
                sz,
                blastoff::MatrixOp::None,
                &beta_gpu,
                &mat_a_gpu,
                sz,
                blastoff::MatrixOp::None,
                &mut mat_c_gpu,
                sz,
            )?;
            stream.synchronize()?;
        }
        end.record(stream)?;
        beg.synchronize()?;
        end.synchronize()?;

        let mut mat_c_actual = Array::<f32, _>::zeros((sz, sz));
        mat_c_gpu.copy_to(&mut mat_c_actual.as_slice_mut().unwrap())?;
        let duration = end.elapsed_time_f32(&beg)? / (NUM_RUNS as f32);
        println!("cuBLAS {sz}x{sz}: {duration:.4}ms");
        if sz < 1024 {
            assert_gemm_eq(&mat_a, &mat_b, &mat_c, alpha, beta, &mat_c_actual);
        }
    }

    Ok(())
}

/// Runs a GEMM kernel (naive or tiled) for a set of matrix sizes and checks correctness.
///
/// # Arguments
/// * `stream` - CUDA stream to use for kernel launches and memory operations.
/// * `module` - CUDA module containing the kernel.
/// * `gemm_fn` - Function pointer to the GEMM kernel launcher.
/// * `kernel_name` - Name of the kernel for logging.
///
/// This function benchmarks the provided GEMM kernel and checks the result for small matrices.
fn run_gemm_kernel(
    stream: &stream::Stream,
    module: &module::Module,
    gemm_fn: &GemmFn,
    kernel_name: &str,
) -> Result<(), Box<dyn Error>> {
    // Sanity check.
    {
        let mat_a = ndarray::arr2::<f32, 2>(&[[1.0, 2.0], [3.0, 4.0]]);
        let mat_b = ndarray::arr2::<f32, 2>(&[[5.0, 6.0], [7.0, 8.0]]);
        let mat_c_expect = ndarray::arr2::<f32, 2>(&[[19.0, 22.0], [43.0, 50.0]]);
        let (alpha, beta) = (1.0f32, 0.0f32);

        let mat_a_gpu = mat_a.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mat_b_gpu = mat_b.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mut mat_c_gpu = unsafe { cust::memory::DeviceBuffer::uninitialized(2 * 2)? };
        stream.synchronize()?;

        gemm_fn(
            stream,
            module,
            &mat_a_gpu,
            &mat_b_gpu,
            &mut mat_c_gpu,
            2,
            2,
            2,
            alpha,
            beta,
        )?;
        stream.synchronize()?;

        let mut mat_c_actual = Array::<f32, _>::zeros((2, 2));
        mat_c_gpu.copy_to(&mut mat_c_actual.as_slice_mut().unwrap())?;
        assert!(mat_c_expect.relative_eq(&mat_c_actual, EPS, EPS));
    }

    for sz in MAT_SIZES.iter().cloned() {
        let mat_a = ndarray::Array2::<f32>::random((sz, sz), Uniform::new(-10., 10.));
        let mat_b = ndarray::Array2::<f32>::random((sz, sz), Uniform::new(-10., 10.));
        let mat_c = ndarray::Array2::<f32>::random((sz, sz), Uniform::new(-10., 10.));
        let (alpha, beta) = (1.0f32, 0.0f32);

        let mat_a_gpu = mat_a.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mat_b_gpu = mat_b.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let mut mat_c_gpu = mat_c.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        stream.synchronize()?;

        // Warm up before timing.
        for _ in 0..NUM_WARMUPS {
            gemm_fn(
                stream,
                module,
                &mat_a_gpu,
                &mat_b_gpu,
                &mut mat_c_gpu,
                sz,
                sz,
                sz,
                alpha,
                beta,
            )?;
        }
        stream.synchronize()?;

        // Time the kernel execution.
        let beg = event::Event::new(event::EventFlags::DEFAULT)?;
        let end = event::Event::new(event::EventFlags::DEFAULT)?;
        beg.record(stream)?;
        for _ in 0..NUM_RUNS {
            gemm_fn(
                stream,
                module,
                &mat_a_gpu,
                &mat_b_gpu,
                &mut mat_c_gpu,
                sz,
                sz,
                sz,
                alpha,
                beta,
            )?;
            stream.synchronize()?;
        }
        end.record(stream)?;
        beg.synchronize()?;
        end.synchronize()?;

        let mut mat_c_actual = Array::<f32, _>::zeros((sz, sz));
        mat_c_gpu.copy_to(&mut mat_c_actual.as_slice_mut().unwrap())?;
        let duration = end.elapsed_time_f32(&beg)? / (NUM_RUNS as f32);
        println!("{kernel_name} {sz}x{sz}: {duration:.4}ms");
        if sz < 1024 {
            assert_gemm_eq(&mat_a, &mat_b, &mat_c, alpha, beta, &mat_c_actual);
        }
    }
    Ok(())
}

/// Asserts that the GEMM result matches the expected value within a tolerance.
///
/// # Arguments
/// * `mat_a` - Left matrix operand.
/// * `mat_b` - Right matrix operand.
/// * `mat_c` - Initial value of the output matrix.
/// * `alpha` - Scalar multiplier for mat_a * mat_b.
/// * `beta` - Scalar multiplier for mat_c.
/// * `mat_c_actual` - Result from the device.
fn assert_gemm_eq<T>(
    mat_a: &ndarray::Array2<T>,
    mat_b: &ndarray::Array2<T>,
    mat_c: &ndarray::Array2<T>,
    alpha: T,
    beta: T,
    mat_c_actual: &ndarray::Array2<T>,
) where
    T: Clone,
    f64: From<T>,
{
    let mat_a = mat_a.mapv(|v| Into::<f64>::into(v));
    let mat_b = mat_b.mapv(|v| Into::<f64>::into(v));
    let mat_c = mat_c.mapv(|v| Into::<f64>::into(v));
    let mat_c_actual = mat_c_actual.mapv(|v| Into::<f64>::into(v));
    let alpha: f64 = alpha.into();
    let beta: f64 = beta.into();
    let mat_c_expect = alpha * mat_a.dot(&mat_b) + beta * mat_c;
    let ok = mat_c_expect.relative_eq(&mat_c_actual, EPS.into(), EPS.into());
    if !ok {
        println!("Actual: {mat_c_actual:?}");
        println!("Expect: {mat_c_expect:?}");
        panic!("GEMM result mismatch");
    }
}

/// Launches the naive GEMM kernel on the device.
///
/// # Arguments
/// * `stream` - CUDA stream to use for kernel launch.
/// * `module` - CUDA module containing the kernel.
/// * `mat_a` - Device buffer for left matrix operand (m x k).
/// * `mat_b` - Device buffer for right matrix operand (k x n).
/// * `mat_c` - Device buffer for output matrix (m x n).
/// * `m` - Number of rows in mat_a and mat_c.
/// * `n` - Number of columns in mat_b and mat_c.
/// * `k` - Number of columns in mat_a and rows in mat_b.
/// * `alpha` - Scalar multiplier for mat_a * mat_b.
/// * `beta` - Scalar multiplier for mat_c.
///
/// This function configures the launch parameters and invokes the naive GEMM kernel.
#[allow(clippy::too_many_arguments)]
pub fn gemm_naive(
    stream: &stream::Stream,
    module: &module::Module,
    mat_a: &memory::DeviceBuffer<f32>,
    mat_b: &memory::DeviceBuffer<f32>,
    mat_c: &mut memory::DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(mat_a.len(), m * k);
    assert_eq!(mat_b.len(), k * n);
    assert_eq!(mat_c.len(), m * n);

    let kernel_cell = cell::LazyCell::new(|| {
        module
            .get_function("gemm_naive")
            .expect("kernel not found.")
    });
    let kernel = &*kernel_cell;

    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
    let block_size = block_size as usize;
    let (block_size_x, block_size_y) = if block_size > m * n {
        (block_size.div_ceil(m) as u32, m as u32)
    } else {
        (1, block_size as u32)
    };
    let (grid_size_x, grid_size_y) = (
        (m as u32).div_ceil(block_size_x),
        (n as u32).div_ceil(block_size_y),
    );
    unsafe {
        launch!(
            kernel<<<
                (grid_size_x, grid_size_y),
                (block_size_x, block_size_y),
                0,
                stream
            >>>(
                mat_a.as_device_ptr(),
                mat_a.len(),
                mat_b.as_device_ptr(),
                mat_b.len(),
                mat_c.as_device_ptr(),
                m,
                n,
                k,
                alpha,
                beta,
            )
        )?;
    };
    Ok(())
}

/// Launches the tiled GEMM kernel on the device.
///
/// # Arguments
/// * `stream` - CUDA stream to use for kernel launch.
/// * `module` - CUDA module containing the kernel.
/// * `mat_a` - Device buffer for left matrix operand (m x k).
/// * `mat_b` - Device buffer for right matrix operand (k x n).
/// * `mat_c` - Device buffer for output matrix (m x n).
/// * `m` - Number of rows in mat_a and mat_c.
/// * `n` - Number of columns in mat_b and mat_c.
/// * `k` - Number of columns in mat_a and rows in mat_b.
/// * `alpha` - Scalar multiplier for mat_a * mat_b.
/// * `beta` - Scalar multiplier for mat_c.
///
/// This function configures the launch parameters and invokes the tiled GEMM kernel.
#[allow(clippy::too_many_arguments)]
pub fn gemm_tiled(
    stream: &stream::Stream,
    module: &module::Module,
    mat_a: &memory::DeviceBuffer<f32>,
    mat_b: &memory::DeviceBuffer<f32>,
    mat_c: &mut memory::DeviceBuffer<f32>,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), Box<dyn Error>> {
    assert_eq!(mat_a.len(), m * k);
    assert_eq!(mat_b.len(), k * n);
    assert_eq!(mat_c.len(), m * n);

    // These values must be aligned with the kernel code.
    const TILE_SIZE: usize = 16;

    let kernel_cell = cell::LazyCell::new(|| {
        module
            .get_function("gemm_tiled")
            .expect("kernel not found.")
    });
    let kernel = &*kernel_cell;

    let (grid_size_x, grid_size_y) = (n.div_ceil(TILE_SIZE) as u32, m.div_ceil(TILE_SIZE) as u32);
    unsafe {
        launch!(
            kernel<<<
                (grid_size_x, grid_size_y),
                (TILE_SIZE as u32, TILE_SIZE as u32),
                0,
                stream
            >>>(
                mat_a.as_device_ptr(),
                mat_a.len(),
                mat_b.as_device_ptr(),
                mat_b.len(),
                mat_c.as_device_ptr(),
                m,
                n,
                k,
                alpha,
                beta,
            )
        )?;
    };
    Ok(())
}
