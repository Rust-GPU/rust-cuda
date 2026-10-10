use crate::gpu_only;
/// Asynchronously copies one `T` from global to shared memory, caching the source in L1/L2.
///
/// The copy is issued but not guaranteed to be complete when this function returns.
/// Use [`cp_async_commit_group`] and [`cp_async_wait_group`] to synchronize.
///
/// # Size
///
/// `T` must be exactly 4, 8, or 16 bytes. Any other size is a compile-time error.
///
/// # Safety
///
/// - `dst` must point into shared memory.
/// - `src` must point into global memory.
/// - Both pointers must be aligned to `size_of::<T>()`.
/// - The pointed-to memory must be valid for the duration of the copy (until the
///   corresponding [`cp_async_wait_group`] or [`cp_async_wait_all`] returns).
#[gpu_only]
pub unsafe fn cp_async_ca<T>(dst: *mut T, src: *const T) {
    const {
        let size = core::mem::size_of::<T>();
        assert!(
            size == 4 || size == 8 || size == 16,
            "cp_async requires T to be exactly 4, 8, or 16 bytes"
        );
    }
    // cp.async requires dst to be a 32-bit shared memory address.
    // Generic pointers must be explicitly converted: cvta.to.shared (64-bit)
    // then truncated to 32-bit, since shared memory is 32-bit addressable.
    unsafe {
        match core::mem::size_of::<T>() {
            4 => core::arch::asm!(
                "cvta.to.shared.u64 {tmp}, {dst};",
                "cvt.u32.u64 {smem}, {tmp};",
                "cp.async.ca.shared.global [{smem}], [{src}], 4;",
                dst  = in(reg64) dst,
                src  = in(reg64) src,
                tmp  = out(reg64) _,
                smem = out(reg32) _,
            ),
            8 => core::arch::asm!(
                "cvta.to.shared.u64 {tmp}, {dst};",
                "cvt.u32.u64 {smem}, {tmp};",
                "cp.async.ca.shared.global [{smem}], [{src}], 8;",
                dst  = in(reg64) dst,
                src  = in(reg64) src,
                tmp  = out(reg64) _,
                smem = out(reg32) _,
            ),
            16 => core::arch::asm!(
                "cvta.to.shared.u64 {tmp}, {dst};",
                "cvt.u32.u64 {smem}, {tmp};",
                "cp.async.ca.shared.global [{smem}], [{src}], 16;",
                dst  = in(reg64) dst,
                src  = in(reg64) src,
                tmp  = out(reg64) _,
                smem = out(reg32) _,
            ),
            _ => unreachable!(),
        }
    }
}

/// Asynchronously copies one `T` from global to shared memory, caching only in L2
/// (bypasses L1). Only valid for 16-byte types.
///
/// Prefer this over [`cp_async_ca`] for streaming access patterns where the data
/// will not be reused, to avoid polluting L1.
///
/// # Safety
///
/// - `dst` must point into shared memory.
/// - `src` must point into global memory.
/// - Both pointers must be 16-byte aligned.
/// - `T` must be exactly 16 bytes — enforced at compile time.
#[gpu_only]
pub unsafe fn cp_async_cg<T>(dst: *mut T, src: *const T) {
    const {
        assert!(
            core::mem::size_of::<T>() == 16,
            "cp_async_cg requires T to be exactly 16 bytes (.cg cache operator only supports 16-byte copies)"
        );
    }
    unsafe {
        core::arch::asm!(
            "cvta.to.shared.u64 {tmp}, {dst};",
            "cvt.u32.u64 {smem}, {tmp};",
            "cp.async.cg.shared.global [{smem}], [{src}], 16;",
            dst  = in(reg64) dst,
            src  = in(reg64) src,
            tmp  = out(reg64) _,
            smem = out(reg32) _,
        )
    }
}

/// Seals all `cp.async` operations issued since the last `cp_async_commit_group` (or
/// program start) into a named group. Groups are completed in FIFO order.
///
/// Must be called before [`cp_async_wait_group`] to define group boundaries.
#[gpu_only]
pub fn cp_async_commit_group() {
    unsafe { core::arch::asm!("cp.async.commit_group;") }
}

/// Waits until there are at most `N` committed `cp.async` groups still in flight.
///
/// - `N = 0`: waits for all groups — equivalent to [`cp_async_wait_all`].
/// - `N = 1`: waits for all but the most recently committed group, allowing one
///   prefetch to remain in flight while computing.
///
/// Must be paired with [`cp_async_commit_group`] to define which copies belong to
/// each group.
#[gpu_only]
pub fn cp_async_wait_group<const N: u32>() {
    unsafe { core::arch::asm!("cp.async.wait_group {0};", const N) }
}

/// Waits for all outstanding `cp.async` copies to complete.
///
/// Equivalent to `cp_async_wait_group::<0>()`.
#[gpu_only]
pub fn cp_async_wait_all() {
    unsafe { core::arch::asm!("cp.async.wait_all;") }
}
