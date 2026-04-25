use cust::prelude::*;
use cust_raw::driver_sys;
use nanorand::{Rng, WyRand};
use std::error::Error;
use std::ffi::{CStr, CString, c_void};
use std::io::Write;
use std::os::raw::c_uint;
use std::ptr;

/// How many numbers to generate and add together.
const NUMBERS_LEN: usize = 100_000;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn load_ptx_with_log(ptx: &str) -> Result<Module, Box<dyn Error>> {
    let cstr = CString::new(ptx).expect("PTX contains nul bytes");

    // Pre-allocate log buffers so the driver can write its real complaint there.
    const LOG_CAP: usize = 16 * 1024;
    let mut info_log = vec![0u8; LOG_CAP];
    let mut error_log = vec![0u8; LOG_CAP];

    // Driver packs values directly into the *mut c_void slot when the payload fits.
    // LOG_VERBOSE = request detailed log
    // INFO/ERROR_LOG_BUFFER = pointer to buffer
    // *_LOG_BUFFER_SIZE_BYTES = capacity (in), bytes written (out)
    let mut options = [
        driver_sys::CUjit_option::CU_JIT_LOG_VERBOSE,
        driver_sys::CUjit_option::CU_JIT_INFO_LOG_BUFFER,
        driver_sys::CUjit_option::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        driver_sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
        driver_sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    ];
    let mut option_values: [*mut c_void; 5] = [
        std::ptr::dangling_mut::<c_void>(),
        info_log.as_mut_ptr() as *mut c_void,
        LOG_CAP as *mut c_void,
        error_log.as_mut_ptr() as *mut c_void,
        LOG_CAP as *mut c_void,
    ];

    let mut module_ptr: driver_sys::CUmodule = ptr::null_mut();
    let res = unsafe {
        driver_sys::cuModuleLoadDataEx(
            &mut module_ptr,
            cstr.as_ptr() as *const c_void,
            options.len() as c_uint,
            options.as_mut_ptr(),
            option_values.as_mut_ptr(),
        )
    };

    let info_len = option_values[2] as usize;
    let error_len = option_values[4] as usize;
    let info_str = String::from_utf8_lossy(&info_log[..info_len.min(LOG_CAP)]);
    let error_str = String::from_utf8_lossy(&error_log[..error_len.min(LOG_CAP)]);

    if !info_str.trim().is_empty() {
        eprintln!("[vecadd] JIT info log ({info_len} bytes):\n{info_str}");
    }
    if !error_str.trim().is_empty() {
        eprintln!("[vecadd] JIT error log ({error_len} bytes):\n{error_str}");
    }
    eprintln!("[vecadd] cuModuleLoadDataEx raw result code: {:?}", res);

    if res != driver_sys::cudaError_enum::CUDA_SUCCESS {
        unsafe {
            let mut err_cstr: *const std::os::raw::c_char = ptr::null();
            if driver_sys::cuGetErrorString(res, &mut err_cstr)
                == driver_sys::cudaError_enum::CUDA_SUCCESS
                && !err_cstr.is_null()
            {
                let msg = CStr::from_ptr(err_cstr).to_string_lossy();
                eprintln!("[vecadd] cuGetErrorString: {msg}");
            }
        }
        return Err(format!("cuModuleLoadDataEx failed: {:?}", res).into());
    }

    // The driver accepted the PTX; drop our raw handle and re-load via cust so the
    // caller gets a typed Module with cust's lifetime/drop machinery.
    let _ = unsafe { driver_sys::cuModuleUnload(module_ptr) };
    Module::from_ptx(ptx, &[]).map_err(|e| e.into())
}

// Flush stdout after every println so it stays ordered against our eprintln
// traces when the two streams get muxed (e.g. over SSH, where stdout would
// otherwise be block-buffered and dump out-of-order).
macro_rules! sayln {
    ($($arg:tt)*) => {{
        println!($($arg)*);
        let _ = std::io::stdout().flush();
    }};
}

macro_rules! step {
    ($label:expr, $expr:expr) => {{
        eprintln!("[vecadd] {} ...", $label);
        match $expr {
            Ok(v) => {
                eprintln!("[vecadd] {} ok", $label);
                v
            }
            Err(e) => {
                eprintln!("[vecadd] {} FAILED: {:?}", $label, e);
                return Err(e.into());
            }
        }
    }};
}

fn main() -> Result<(), Box<dyn Error>> {
    // generate our random vectors.
    let mut wyrand = WyRand::new();
    let mut lhs = vec![2.0f32; NUMBERS_LEN];
    wyrand.fill(&mut lhs);
    let mut rhs = vec![0.0f32; NUMBERS_LEN];
    wyrand.fill(&mut rhs);

    let _ctx = step!("cust::quick_init", cust::quick_init());

    let (driver_major, driver_minor) = step!(
        "CudaApiVersion::get",
        cust::CudaApiVersion::get().map(|v| (v.major(), v.minor()))
    );
    eprintln!("[vecadd] CUDA driver API version: {driver_major}.{driver_minor}");

    let device = step!("Device::get_device(0)", cust::device::Device::get_device(0));
    let cc_major = step!(
        "Device::get_attribute(ComputeCapabilityMajor)",
        device.get_attribute(cust::device::DeviceAttribute::ComputeCapabilityMajor)
    );
    let cc_minor = step!(
        "Device::get_attribute(ComputeCapabilityMinor)",
        device.get_attribute(cust::device::DeviceAttribute::ComputeCapabilityMinor)
    );
    let name = step!("Device::name", device.name());
    eprintln!("[vecadd] GPU: {name} (compute {cc_major}.{cc_minor})");

    eprintln!("[vecadd] PTX size: {} bytes", PTX.len());
    eprintln!(
        "[vecadd] PTX header: {}",
        PTX.lines().take(10).collect::<Vec<_>>().join(" | ")
    );

    // Load PTX via raw cuModuleLoadDataEx so we can capture the JIT error/info log
    // buffers; cust's ModuleJitOption doesn't surface those yet, and on UnknownError
    // the log is the only way to see the driver's real complaint.
    let module = step!(
        "cuModuleLoadDataEx (with JIT log buffers)",
        load_ptx_with_log(PTX)
    );

    let stream = step!("Stream::new", Stream::new(StreamFlags::NON_BLOCKING, None));

    let lhs_gpu = step!("DeviceBuffer::from lhs", lhs.as_slice().as_dbuf());
    let rhs_gpu = step!("DeviceBuffer::from rhs", rhs.as_slice().as_dbuf());

    let mut out = vec![0.0f32; NUMBERS_LEN];
    let out_buf = step!("DeviceBuffer::from out", out.as_slice().as_dbuf());

    let vecadd = step!(
        "Module::get_function(\"vecadd\")",
        module.get_function("vecadd")
    );

    let (_, block_size) = step!(
        "suggested_launch_configuration",
        vecadd.suggested_launch_configuration(0, 0.into())
    );

    let grid_size = (NUMBERS_LEN as u32).div_ceil(block_size);

    sayln!("using {grid_size} blocks and {block_size} threads per block");

    eprintln!("[vecadd] launching kernel ...");
    unsafe {
        launch!(
            vecadd<<<grid_size, block_size, 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                lhs_gpu.len(),
                rhs_gpu.as_device_ptr(),
                rhs_gpu.len(),
                out_buf.as_device_ptr(),
            )
        )
        .map_err(|e| {
            eprintln!("[vecadd] launch FAILED: {e:?}");
            e
        })?;
    }
    eprintln!("[vecadd] launch queued ok");

    step!("stream.synchronize", stream.synchronize());

    step!("copy_to", out_buf.copy_to(&mut out));

    sayln!("{} + {} = {}", lhs[0], rhs[0], out[0]);

    Ok(())
}
