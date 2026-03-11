//! CUDA-specific pointer handling logic.

use crate::gpu_only;
#[cfg(target_os = "cuda")]
use core::arch::asm;

/// Special areas of GPU memory where a pointer could reside.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressSpace {
    /// Memory available for reading and writing to the entire device.
    Global,
    /// Block-local read/write memory available to all threads in a block.
    Shared,
    /// Read-only memory available to the whole device.
    Constant,
    /// Thread-local read/write memory only available to an individual thread.
    Local,
}

/// Determines whether a pointer is in a specific address space.
///
/// # Safety
///
/// The pointer must be valid for an instance of `T`, otherwise Undefined Behavior is exhibited.
// TODO(RDambrosio016): Investigate subpar codegen for this function. It seems nvcc implements this not using
// inline asm, but instead with some sort of compiler intrinsic, because its able to optimize away the function
// a lot of the time.
#[gpu_only]
pub unsafe fn is_in_address_space<T>(ptr: *const T, address_space: AddressSpace) -> bool {
    let ret: u32;
    unsafe {
        // Create a predicate register to store the result of the isspacep into.
        asm!(".reg .pred p;");

        // Perform the actual isspacep operation, and store the result in the predicate register we
        // made.
        match address_space {
            AddressSpace::Global => asm!("isspacep.global p, {}", in(reg64) ptr),
            AddressSpace::Shared => asm!("isspacep.shared p, {}", in(reg64) ptr),
            AddressSpace::Constant => asm!("isspacep.const p, {}", in(reg64) ptr),
            AddressSpace::Local => asm!("isspacep.local p, {}", in(reg64) ptr),
        }

        // Finally, use the predicate register to write out a value.
        asm!("selp.u32 {}, 1, 0, p;", out(reg32) ret);
    }

    ret != 0
}

/// Converts a pointer from a generic address space, to a specific address space.
/// This maps directly to the [`cvta`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta) PTX instruction.
///
/// # Safety
///
/// The pointer must be valid for an instance of `T`, and the pointer must fall in the specific address space in memory,
/// otherwise Undefined Behavior is exhibited.
#[gpu_only]
pub unsafe fn convert_generic_to_specific_address_space<T>(
    ptr: *const T,
    address_space: AddressSpace,
) -> *const T {
    let ret: *const T;

    unsafe {
        match address_space {
            AddressSpace::Global => asm!(
                "cvta.to.global.u64 {}, {}",
                out(reg64) ret,
                in(reg64) ptr
            ),
            AddressSpace::Shared => asm!(
                "cvta.to.shared.u64 {}, {}",
                out(reg64) ret,
                in(reg64) ptr
            ),
            AddressSpace::Constant => asm!(
                "cvta.to.const.u64 {}, {}",
                out(reg64) ret,
                in(reg64) ptr
            ),
            AddressSpace::Local => asm!(
                "cvta.to.local.u64 {}, {}",
                out(reg64) ret,
                in(reg64) ptr
            ),
        }
    }

    ret
}

/// Converts a pointer in a specific address space, to a generic address space.
/// This maps directly to the [`cvta`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta) PTX instruction.
///
/// # Safety
///
/// The pointer must be valid for an instance of `T`, and the pointer must fall in the specific address space in memory,
/// otherwise Undefined Behavior is exhibited.
#[gpu_only]
pub unsafe fn convert_specific_address_space_to_generic<T>(
    ptr: *const T,
    address_space: AddressSpace,
) -> *const T {
    let ret: *const T;

    unsafe {
        match address_space {
            AddressSpace::Global => asm!(
                "cvta.global.u64 {}, {}",
                out(reg64) ret,
                in(reg64) ptr
            ),
            AddressSpace::Shared => asm!(
                "cvta.shared.u64 {}, {}",
                out(reg64) ret,
                in(reg64) ptr
            ),
            AddressSpace::Constant => asm!(
                "cvta.const.u64 {}, {}",
                out(reg64) ret,
                in(reg64) ptr
            ),
            AddressSpace::Local => asm!(
                "cvta.local.u64 {}, {}",
                out(reg64) ret,
                in(reg64) ptr
            ),
        }
    }

    ret
}
