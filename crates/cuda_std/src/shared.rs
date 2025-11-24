//! Dynamic shared memory handling.
//!
//! Static shared memory is done via `#[address_space(shared)] static mut ...;`.

use crate::gpu_only;

/// Gets a pointer to the dynamic shared memory that was allocated by the caller of the kernel. The
/// data is left uninitialized.
///
/// **Calling this function multiple times will yield the same pointer**.
#[gpu_only]
pub fn dynamic_shared_mem<T>() -> *mut T {
    // it is unclear whether an alignment of 16 is actually required for correctness, however,
    // it seems like nvcc always generates the global with .align 16 no matter the type, so we just copy
    // nvcc's behavior for now.
    extern "C" {
        // need to use nvvm_internal and not address_space because address_space only parses
        // static definitions, not extern static definitions.
        #[nvvm_internal::addrspace(3)]
        #[allow(improper_ctypes)]
        // mangle it a bit to make sure nobody makes the same thing
        #[link_name = "_Zcuda_std_dyn_shared"]
        static DYN_SHARED: ::core::cell::UnsafeCell<u128>;
    }

    // SAFETY: extern statics is how dynamic shared mem is done in CUDA. This will turn into
    // an extern variable decl in ptx, which is the same thing nvcc does if you dump the ptx from a cuda file.
    unsafe { DYN_SHARED.get() as *mut T }
}
