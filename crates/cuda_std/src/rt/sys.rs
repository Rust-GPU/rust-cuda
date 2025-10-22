//! Raw bindings to cuda_device_runtime_api functions.

use core::ffi::c_void;

#[allow(non_camel_case_types)]
pub type c_char = i8;
#[allow(non_camel_case_types)]
pub type c_int = i32;
#[allow(non_camel_case_types)]
pub type c_uint = u32;
pub use crate::rt::driver_types_sys::*;

// TODO(RDambrosio016): We should probably create a common crate
// to share this stuff with cust.

extern "C" {
    pub fn cudaDeviceGetAttribute(
        value: *mut c_int,
        attr: cudaDeviceAttr,
        device: c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;
    pub fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaPeekAtLastError() -> cudaError_t;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
    pub fn cudaGetErrorName(error: cudaError_t) -> *const c_char;
    pub fn cudaGetDeviceCount(count: *mut c_int) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
    pub fn cudaStreamCreateWithFlags(pStream: *mut cudaStream_t, flags: c_uint) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamWaitEvent(
        stream: cudaStream_t,
        event: cudaEvent_t,
        flags: c_uint,
    ) -> cudaError_t;
    pub fn cudaEventCreateWithFlags(event: *mut cudaEvent_t, flags: c_uint) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventRecordWithFlags(
        event: cudaEvent_t,
        stream: cudaStream_t,
        flags: c_uint,
    ) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;

    pub fn cudaGetParameterBufferV2(
        func: *const c_void,
        gridDimension: dim3,
        block_dimension: dim3,
        shared_mem_size: c_uint,
    ) -> *mut c_void;
    pub fn cudaLaunchDeviceV2(parameter_buffer: *mut c_void, stream: cudaStream_t) -> cudaError_t;

    pub fn cudaIpcGetMemHandle(handle: *mut cudaIpcMemHandle_t, dev_ptr: *mut c_void) -> cudaError_t;
    pub fn cudaIpcOpenMemHandle(dev_ptr: *mut *mut c_void, handle: cudaIpcMemHandle_t, flags: u32) -> cudaError_t;
    pub fn cudaIpcCloseMemHandle(dev_ptr: *mut c_void) -> cudaError_t;
}
