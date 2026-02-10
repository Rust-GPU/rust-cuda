//! Functions that work over warps of threads.
//!
//! Warps in CUDA are groups of 32 threads that are dispatched together inside of
//! thread blocks and execute in SIMT fashion.

use crate::gpu_only;
#[cfg(target_os = "cuda")]
use core::arch::asm;
use half::{bf16, f16};

/// Synchronizes all of the threads inside of this warp according to `mask`.
///
/// # Safety
///
/// The behavior of this function is undefined if:
/// - Any thread inside `mask` has exited.
/// - The executing thread is not inside of `mask`.
///
/// Moreover, on compute_62 and below, all of the threads inside `mask` must call
/// `sync` with the __exact same__ mask. Otherwise it is undefined behavior.
#[gpu_only]
#[inline(always)]
pub unsafe fn sync_warp(mask: u32) {
    unsafe extern "C" {
        #[link_name = "llvm.nvvm.bar.warp.sync"]
        fn sync(mask: u32);
    }

    unsafe {
        sync(mask);
    }
}

/// Returns the thread's lane within its warp. This value ranges from `0` to `WARP_SIZE - 1` (`WARP_SIZE` is 32 on all
/// architectures currently).
#[gpu_only]
#[inline(always)]
pub fn lane_id() -> u32 {
    let mut out;
    unsafe {
        asm!(
            "mov.u32 {}, %laneid;",
            out(reg32) out
        );
    }
    out
}

/// Queries a mask of the active threads in the warp.
#[gpu_only]
#[inline(always)]
pub fn activemask() -> u32 {
    let mut out;
    unsafe {
        asm!(
            "activemask.b32 {};",
            out(reg32) out
        );
    }
    out
}

/// Synchronizes threads in a warp then performs a reduction operation.
///
/// `mask` is a bitmask indicating which threads in the warp should participate in the reduction.
/// The threads in `mask` will synchronize then perform the reduction op indicated by `op` and return the reduced value.
///
/// This intrinsic is only available on Compute Capabilities 8.x (Ampere) and above.
///
/// # Safety
///
/// Behavior is undefined if any thread specified inside of `mask` has exited. Additionally, every
/// thread must execute the function with the same mask.
#[gpu_only]
#[inline(always)]
pub unsafe fn warp_reduce<T: WarpReduceValue>(mask: u32, value: T, op: WarpReductionOp) -> T {
    unsafe { T::reduce(mask, value, op) }
}

/// The type of operation to apply in a warp reduction.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarpReductionOp {
    Add,
    Min,
    Max,
    And,
    Or,
    Xor,
}

pub trait WarpReduceValue: Sized {
    #[allow(clippy::missing_safety_doc)]
    unsafe fn reduce(mask: u32, value: Self, op: WarpReductionOp) -> Self;
}

macro_rules! impl_reduce {
    ($($type:ty),* $(,)?) => {
        $(
            paste::paste! {
                impl WarpReduceValue for $type {
                    unsafe fn reduce(mask: u32, value: Self, op: WarpReductionOp) -> Self {
                        unsafe { [<warp_reduce_ $type>](mask, value, op) }
                    }
                }
            }
        )*
    }
}

impl_reduce! {
    i32,
    u32,
}

#[gpu_only]
unsafe fn warp_reduce_32(mask: u32, value: u32, op: WarpReductionOp) -> u32 {
    let out;
    unsafe {
        match op {
            WarpReductionOp::And => {
                asm!(
                    "redux.sync.and.b32 {}, {}, {};",
                    out(reg32) out,
                    in(reg32) value,
                    in(reg32) mask
                );
            }
            WarpReductionOp::Or => {
                asm!(
                    "redux.sync.or.b32 {}, {}, {};",
                    out(reg32) out,
                    in(reg32) value,
                    in(reg32) mask
                );
            }
            WarpReductionOp::Xor => {
                asm!(
                    "redux.sync.xor.b32 {}, {}, {};",
                    out(reg32) out,
                    in(reg32) value,
                    in(reg32) mask
                );
            }
            _ => unreachable!(),
        }
    }
    out
}

#[gpu_only]
unsafe fn warp_reduce_u32(mask: u32, value: u32, op: WarpReductionOp) -> u32 {
    let out;
    unsafe {
        match op {
            WarpReductionOp::Add => {
                asm!(
                    "redux.sync.add.u32 {}, {}, {};",
                    out(reg32) out,
                    in(reg32) value,
                    in(reg32) mask
                );
            }
            WarpReductionOp::Min => {
                asm!(
                    "redux.sync.min.u32 {}, {}, {};",
                    out(reg32) out,
                    in(reg32) value,
                    in(reg32) mask
                );
            }
            WarpReductionOp::Max => {
                asm!(
                    "redux.sync.max.u32 {}, {}, {};",
                    out(reg32) out,
                    in(reg32) value,
                    in(reg32) mask
                );
            }
            _ => out = warp_reduce_32(mask, value, op),
        }
    }
    out
}

#[gpu_only]
unsafe fn warp_reduce_i32(mask: u32, value: i32, op: WarpReductionOp) -> i32 {
    let out;
    unsafe {
        match op {
            WarpReductionOp::Add => {
                asm!(
                    "redux.sync.add.s32 {}, {}, {};",
                    out(reg32) out,
                    in(reg32) value,
                    in(reg32) mask
                );
            }
            WarpReductionOp::Min => {
                asm!(
                    "redux.sync.min.s32 {}, {}, {};",
                    out(reg32) out,
                    in(reg32) value,
                    in(reg32) mask
                );
            }
            WarpReductionOp::Max => {
                asm!(
                    "redux.sync.max.s32 {}, {}, {};",
                    out(reg32) out,
                    in(reg32) value,
                    in(reg32) mask
                );
            }
            _ => out = warp_reduce_32(mask, value as u32, op) as i32,
        }
    }
    out
}

/// Synchronizes threads in a warp and performs a broadcast-and-compare operation between them.
///
/// `mask` is a bitmask dictating what threads should participate in the operation. All the threads in
/// `mask` will synchronize then perform a warp match operation. The result is a bitmask of all the threads
/// which have the same `value`.
///
/// This intrinsic is only available on Compute Capabilities 7.x (Volta) and above.
///
/// # Safety
///
/// Behavior is undefined if any thread specified inside of `mask` has exited. Additionally, every
/// thread must execute the function with the same mask.
#[gpu_only]
#[inline(always)]
pub unsafe fn warp_match_any<T: WarpMatchValue>(mask: u32, value: T) -> u32 {
    unsafe { T::match_any(mask, value) }
}

/// Synchronizes threads in a warp and performs a broadcast-and-compare operation between them.
///
/// `mask` is a bitmask dictating what threads should participate in the operation. All the threads in
/// `mask` will synchronize then perform a warp match operation. Returns `Some(mask)` if all threads in `mask` have
/// the same value for `value`, otherwise returns `None`.
///
/// This intrinsic is only available on Compute Capabilities 7.x (Volta) and above.
///
/// # Safety
///
/// Behavior is undefined if any thread specified inside of `mask` has exited. Additionally, every
/// thread must execute the function with the same mask.
#[gpu_only]
#[inline(always)]
pub unsafe fn warp_match_all<T: WarpMatchValue>(mask: u32, value: T) -> Option<u32> {
    unsafe { T::match_all(mask, value) }
}

/// A value that can be used inside of a warp match.
pub trait WarpMatchValue: Sized {
    #[allow(clippy::missing_safety_doc)]
    unsafe fn match_any(mask: u32, value: Self) -> u32;
    #[allow(clippy::missing_safety_doc)]
    unsafe fn match_all(mask: u32, value: Self) -> Option<u32>;
}

macro_rules! impl_match {
    ($($type:ty, $width:literal),* $(,)?) => {
        $(
            paste::paste! {
                impl WarpMatchValue for $type {
                    unsafe fn match_any(mask: u32, value: Self) -> u32 {
                        unsafe { [<match_any_ $width>](mask, value as [<u $width>]) }
                    }
                    unsafe fn match_all(mask: u32, value: Self) -> Option<u32> {
                        let (val, pred) = unsafe { [<match_all_ $width>](mask, value as [<u $width>]) };
                        pred.then(|| val)
                    }
                }
            }
        )*
    }
}

impl_match! {
    i32, 32,
    i64, 64,
    u32, 32,
    u64, 64,
    f32, 32,
    f64, 64,
}

#[gpu_only]
#[inline(always)]
unsafe fn match_any_32(mask: u32, value: u32) -> u32 {
    unsafe extern "C" {
        #[link_name = "llvm.nvvm.match.any.sync.i32"]
        fn __nvvm_warp_match_any_32(mask: u32, value: u32) -> u32;
    }
    unsafe { __nvvm_warp_match_any_32(mask, value) }
}

#[gpu_only]
#[inline(always)]
unsafe fn match_any_64(mask: u32, value: u64) -> u32 {
    unsafe extern "C" {
        #[link_name = "llvm.nvvm.match.any.sync.i64"]
        fn __nvvm_warp_match_any_64(mask: u32, value: u64) -> u32;
    }
    unsafe { __nvvm_warp_match_any_64(mask, value) }
}

#[gpu_only]
#[inline(always)]
unsafe fn match_all_32(mask: u32, value: u32) -> (u32, bool) {
    unsafe extern "C" {
        #[allow(improper_ctypes)]
        fn __nvvm_warp_match_all_32(mask: u32, value: u32) -> (u32, bool);
    }
    unsafe { __nvvm_warp_match_all_32(mask, value) }
}

#[gpu_only]
#[inline(always)]
unsafe fn match_all_64(mask: u32, value: u64) -> (u32, bool) {
    unsafe extern "C" {
        #[allow(improper_ctypes)]
        fn __nvvm_warp_match_all_64(mask: u32, value: u64) -> (u32, bool);
    }
    unsafe { __nvvm_warp_match_all_64(mask, value) }
}

/// Synchronizes a subset of threads in a warp then performs a reduce-and-broadcast
/// operation, returning `true` only if `predicate` evaluates to `true` for all threads
/// in the warp that are counted in `mask`. Mask is usually [`u32::MAX`].
///
/// # Safety
///
/// Behavior is undefined if:
/// - Any thread participating in the vote has exited or the executing thread is not in
///   `mask`.
/// - For `compute_62` and below, all threads in `mask` must call this function in
///   convergence, and only threads belonging to the `mask` can be active when the
///   intrinsic is called.
/// - A thread tries to execute this function while not being present in `mask`.
#[gpu_only]
pub unsafe fn warp_vote_all(mask: u32, predicate: bool) -> bool {
    let mut out: u32;
    unsafe {
        asm!(
            "{{",
            ".reg .pred %p<3>;",
            "setp.eq.u32 %p1, {}, 1;",
            "vote.sync.all.pred %p2, %p1, {};",
            "selp.u32 {}, 0, 1, %p2;",
            "}}",
            in(reg32) predicate as u32,
            in(reg32) mask,
            out(reg32) out
        );
    }
    out != 0
}

/// Synchronizes a subset of threads in a warp then performs a reduce-and-broadcast
/// operation, returning `true` only if `predicate` evaluates to `true` for any of the
/// threads in the warp that are counted in `mask`. Mask is usually [`u32::MAX`].
///
/// # Safety
///
/// Behavior is undefined if:
/// - Any thread participating in the vote has exited or the executing thread is not in
///   `mask`.
/// - For `compute_62` and below, all threads in `mask` must call this function in
///   convergence, and only threads belonging to the `mask` can be active when the
///   intrinsic is called.
/// - A thread tries to execute this function while not being present in `mask`.
#[gpu_only]
pub unsafe fn warp_vote_any(mask: u32, predicate: bool) -> bool {
    let mut out: u32;
    unsafe {
        asm!(
            "{{",
            ".reg .pred %p<3>;",
            "setp.eq.u32 %p1, {}, 1;",
            "vote.sync.any.pred %p2, %p1, {};",
            "selp.u32 {}, 0, 1, %p2;",
            "}}",
            in(reg32) predicate as u32,
            in(reg32) mask,
            out(reg32) out
        );
    }
    out != 0
}

/// Synchronizes a subset of threads in a warp then performs a reduce-and-broadcast
/// operation, returning an integer where every Nth bit is set only if the predicate
/// from the Nth thread is `true`. Inactive threads will be counted as a `0` in the mask
/// for that bit. Mask is usually [`u32::MAX`].
///
/// # Safety
///
/// Behavior is undefined if:
/// - Any thread participating in the vote has exited or the executing thread is not in
///   `mask`.
/// - For `compute_62` and below, all threads in `mask` must call this function in
///   convergence, and only threads belonging to the `mask` can be active when the
///   intrinsic is called.
/// - A thread tries to execute this function while not being present in `mask`.
#[gpu_only]
pub unsafe fn warp_vote_ballot(mask: u32, predicate: bool) -> u32 {
    let mut out: u32;
    unsafe {
        asm!(
            "{{",
            ".reg .pred %p1;",
            "setp.eq.u32 %p1, {}, 1;",
            "vote.sync.ballot.b32 {}, %p1, {};",
            "}}",
            in(reg32) predicate as u32,
            out(reg32) out,
            in(reg32) mask,
        );
    }
    out
}

/// Waits for threads in a warp to reach this point and shuffles a value across the
/// threads.
///
/// # Arguments
///
///   - `mask` dictates what threads will participate in the shuffle, usually
///     [`u32::MAX`] to indicate all threads.
///   - `value` is the value that will be shuffled across the threads. i.e. the value
///     that will be given to the thread that calculates this thread as its target lane.
///   - `delta` is the value that will be subtracted from the current thread's lane to
///     calculate the target lane.
///    - `width` dictates how to optionally split the warp into subsections, it must be
///      a power of two and lower than `32`. calculated source lane values will NOT
///      wrap around the value of `width`. Usually just `32`.
///
/// # Returns
///
/// Returns the value from the target lane and a bool indicating if the target lane was
/// active or not.
///
/// # Note
///
/// Shuffles are always 32-bit shuffles, shuffles less than 32 bits will still shuffle a
/// 32-bit value, and shuffles greater than 32 bits will perform `sizeof(T) / 4`
/// shuffles. Therefore it is better to pack multiple 16 or 8 bit values into one value
/// if you are doing multiple shuffles.
///
/// # Panics
///
/// Panics if `width` is not a power of two or higher than 32.
///
/// # Safety
///
/// Behavior is undefined if:
///   - Any thread participating in the shuffle has exited or the executing thread is
///     not in `mask`.
///   - For `compute_62` and below, all threads in `mask` must call the same function in
///     convergence, and only the threads in `mask` can be active when the shuffle is
///     called.
///
/// The returned value returned is unspecified if the calculated target lane is
/// inactive.
pub unsafe fn warp_shuffle_down<T: WarpShuffleValue>(
    mask: u32,
    value: T,
    delta: u32,
    width: u32,
) -> (T, bool) {
    unsafe { T::shuffle(WarpShuffleMode::Down, mask, value, delta, width) }
}

/// Waits for threads in a warp to reach this point and shuffles a value across the
/// threads.
///
/// # Arguments
///
///   - `mask` dictates what threads will participate in the shuffle, usually
///     [`u32::MAX`] to indicate all threads.
///   - `value` is the value that will be shuffled across the threads. i.e. the value
///     that will be given to the thread that calculates this thread as its target lane.
///   - `delta` is the value that will be added to the current thread's lane to
///     calculate the target lane.
///   - `width` dictates how to optionally split the warp into subsections, it must be a
///     power of two and lower than `32`. calculated source lane values will NOT wrap
///     around the value of `width`. Usually just `32`.
///
/// # Returns
///
/// Returns the value from the target lane and a bool indicating if the target lane was
/// active or not.
///
/// # Note
///
/// Shuffles are always 32-bit shuffles, shuffles less than 32 bits will still shuffle a
/// 32-bit value, and shuffles greater than 32 bits will perform `sizeof(T) / 4`
/// shuffles. Therefore it is better to pack multiple 16 or 8 bit values into one value
/// if you are doing multiple shuffles.
///
/// # Panics
///
/// Panics if `width` is not a power of two or higher than 32.
///
/// # Safety
///
/// Behavior is undefined if:
///   - Any thread participating in the shuffle has exited or the executing thread is
///     not in `mask`.
///   - For `compute_62` and below, all threads in `mask` must call the same function in
///     convergence, and only the threads in `mask` can be active when the shuffle is
///     called.
///
/// The returned value returned is unspecified if the calculated target lane is
/// inactive.
pub unsafe fn warp_shuffle_up<T: WarpShuffleValue>(
    mask: u32,
    value: T,
    delta: u32,
    width: u32,
) -> (T, bool) {
    unsafe { T::shuffle(WarpShuffleMode::Up, mask, value, delta, width) }
}

/// Waits for threads in a warp to reach this point and shuffles a value across the
/// threads.
///
/// # Arguments
///
///   - `mask` dictates what threads will participate in the shuffle, usually
///     [`u32::MAX`] to indicate all threads.
///   - `value` is the value that will be shuffled across the threads. i.e. the value
///     that will be given to the thread that calculates this thread as its target
///     lane.
///   - `idx` is the target lane that will be used as the source of this thread's
///     returned value.
///   - `width` dictates how to optionally split the warp into subsections, it must be a
///     power of two and lower than `32`. calculated source lane values will NOT wrap
///     around the value of `width`. Usually just `32`.
///
/// # Returns
///
/// Returns the value from the target lane and a bool indicating if the target lane was
/// active or not.
///
/// # Note
///
/// Shuffles are always 32-bit shuffles, shuffles less than 32 bits will still shuffle a
/// 32-bit value, and shuffles greater than 32 bits will perform `sizeof(T) / 4`
/// shuffles. Therefore it is better to pack multiple 16 or 8 bit values into one value
/// if you are doing multiple shuffles.
///
/// # Panics
///
/// Panics if `width` is not a power of two or higher than 32.
///
/// # Safety
///
/// Behavior is undefined if:
///   - Any thread participating in the shuffle has exited or the executing thread is
///     not in `mask`.
///   - For `compute_62` and below, all threads in `mask` must call the same function in
///     convergence, and only the threads in `mask` can be active when the shuffle is
///     called.
///
/// The returned value returned is unspecified if the calculated target lane is
/// inactive.
pub unsafe fn warp_shuffle_idx<T: WarpShuffleValue>(
    mask: u32,
    value: T,
    idx: u32,
    width: u32,
) -> (T, bool) {
    unsafe { T::shuffle(WarpShuffleMode::Idx, mask, value, idx, width) }
}

/// Waits for threads in a warp to reach this point and shuffles a value across the
/// threads.
///
/// # Arguments
///
///   - `mask` dictates what threads will participate in the shuffle, usually
///     [`u32::MAX`] to indicate all threads.
///   - `value` is the value that will be shuffled across the threads. i.e. the value
///     that will be given to the thread that calculates this thread as its target lane.
///   - `lane_mask` is the value that will be XOR'd by the current thread's lane id to
///     calculate the target lane. i.e. the target lane will be `lane_id ^ lane_mask`.
///    - `width` dictates how to optionally split the warp into subsections, it must be
///      a power of two and lower than `32`. calculated source lane values will NOT wrap
///      around the value of `width`. Usually just `32`.
///
/// # Returns
///
/// Returns the value from the target lane and a bool indicating if the target lane was
/// active or not.
///
/// # Note
///
/// Shuffles are always 32-bit shuffles, shuffles less than 32 bits will still shuffle a
/// 32-bit value, and shuffles greater than 32 bits will perform `sizeof(T) / 4`
/// shuffles. Therefore it is better to pack multiple 16 or 8 bit values into one value
/// if you are doing multiple shuffles.
///
/// # Panics
///
/// Panics if `width` is not a power of two or higher than 32.
///
/// # Safety
///
/// Behavior is undefined if:
///   - Any thread participating in the shuffle has exited or the executing thread is
///     not in `mask`.
///   - For `compute_62` and below, all threads in `mask` must call the same function in
///     convergence, and only the threads in `mask` can be active when the shuffle is
///     called.
///
/// The returned value returned is unspecified if the calculated target lane is
/// inactive.
pub unsafe fn warp_shuffle_xor<T: WarpShuffleValue>(
    mask: u32,
    value: T,
    lane_mask: u32,
    width: u32,
) -> (T, bool) {
    unsafe { T::shuffle(WarpShuffleMode::Xor, mask, value, lane_mask, width) }
}

/// A value that can be used in a warp shuffle
pub trait WarpShuffleValue: Sized {
    /// Executes the shuffle, note that `mode` must be a constant value.
    #[allow(clippy::missing_safety_doc)]
    unsafe fn shuffle(
        mode: WarpShuffleMode,
        mask: u32,
        value: Self,
        b: u32,
        width: u32,
    ) -> (Self, bool);
}

macro_rules! impl_shuffle {
    ($($type:ty, $width:literal),*, $(,)?) => {
        $(
            paste::paste! {
                impl WarpShuffleValue for $type {
                    unsafe fn shuffle(
                        mode: WarpShuffleMode,
                        mask: u32,
                        value: Self,
                        b: u32,
                        width: u32,
                    ) -> (Self, bool) {
                        let (res, oob) = unsafe { [<warp_shuffle_ $width>](mode, mask, value as [<u $width>], b, width) };
                        (res as $type, oob)
                    }
                }
            }
        )*
    };
}

impl_shuffle! {
    i8, 8,
    i16, 16,
    i32, 32,
    i64, 64,
    i128, 128,
    u8, 8,
    u16, 16,
    u32, 32,
    u64, 64,
    u128, 128,
}

// special cases

impl WarpShuffleValue for f32 {
    unsafe fn shuffle(
        mode: WarpShuffleMode,
        mask: u32,
        value: Self,
        b: u32,
        width: u32,
    ) -> (Self, bool) {
        let (res, oob) = unsafe { warp_shuffle_32(mode, mask, value.to_bits(), b, width) };
        (f32::from_bits(res), oob)
    }
}

impl WarpShuffleValue for f64 {
    unsafe fn shuffle(
        mode: WarpShuffleMode,
        mask: u32,
        value: Self,
        b: u32,
        width: u32,
    ) -> (Self, bool) {
        let (res, oob) = unsafe { warp_shuffle_64(mode, mask, value.to_bits(), b, width) };
        (f64::from_bits(res), oob)
    }
}

impl WarpShuffleValue for f16 {
    unsafe fn shuffle(
        mode: WarpShuffleMode,
        mask: u32,
        value: Self,
        b: u32,
        width: u32,
    ) -> (Self, bool) {
        let (res, oob) = unsafe { warp_shuffle_16(mode, mask, value.to_bits(), b, width) };
        (f16::from_bits(res), oob)
    }
}

impl WarpShuffleValue for bf16 {
    unsafe fn shuffle(
        mode: WarpShuffleMode,
        mask: u32,
        value: Self,
        b: u32,
        width: u32,
    ) -> (Self, bool) {
        let (res, oob) = unsafe { warp_shuffle_16(mode, mask, value.to_bits(), b, width) };
        (bf16::from_bits(res), oob)
    }
}

#[doc(hidden)]
#[repr(u32)]
#[derive(Clone, Copy)]
pub enum WarpShuffleMode {
    Idx = 0,
    Up = 1,
    Down = 2,
    Xor = 3,
}

// C-compatible struct to match LLVM IR's {i32, i8} return type
// This fixes an ABI mismatch where Rust would represent (u32, bool) as [2 x i32]
// but the LLVM intrinsic returns {i32, i8} (a struct, not an array)
#[doc(hidden)]
#[repr(C)]
pub struct WarpShuffleResult {
    value: u32,
    predicate: u8,
}

#[gpu_only]
unsafe fn warp_shuffle_32(
    mode: WarpShuffleMode,
    mask: u32,
    value: u32,
    b: u32,
    width: u32,
) -> (u32, bool) {
    unsafe extern "C" {
        // see libintrinsics.ll
        // Returns {i32, i8} in LLVM IR, which maps to our WarpShuffleResult struct
        fn __nvvm_warp_shuffle(mask: u32, mode: u32, a: u32, b: u32, c: u32) -> WarpShuffleResult;
    }

    assert!(
        !(width & (width - 1)) != 0 && width <= 32,
        "width must be a power of 2 and less than or equal to 32"
    );

    // mimicking nvcc's behavior
    let mut c = 0;
    c |= 0b11111;
    c |= (32 - width) << 8;

    let result = unsafe { __nvvm_warp_shuffle(mask, mode as u32, value, b, c) };
    (result.value, result.predicate != 0)
}

unsafe fn warp_shuffle_128(
    mode: WarpShuffleMode,
    mask: u32,
    value: u128,
    b: u32,
    width: u32,
) -> (u128, bool) {
    let first_half = value as u64;
    let second_half = (value >> 64) as u64;
    // shuffle the first and second half of the value then recombine them
    // this will perform 4 shuffles in total (4 32-bit shuffles)
    let (new_first_half, oob) = unsafe { warp_shuffle_64(mode, mask, first_half, b, width) };
    let (new_second_half, _) = unsafe { warp_shuffle_64(mode, mask, second_half, b, width) };
    (
        ((new_second_half as u128) << 64) | (new_first_half as u128),
        oob,
    )
}

unsafe fn warp_shuffle_64(
    mode: WarpShuffleMode,
    mask: u32,
    value: u64,
    b: u32,
    width: u32,
) -> (u64, bool) {
    let first_half = value as u32;
    let second_half = (value >> 32) as u32;
    // shuffle the first and second half of the value then recombine them
    let (new_first_half, oob) = unsafe { warp_shuffle_32(mode, mask, first_half, b, width) };
    let (new_second_half, _) = unsafe { warp_shuffle_32(mode, mask, second_half, b, width) };
    (
        ((new_second_half as u64) << 32) | (new_first_half as u64),
        oob,
    )
}

unsafe fn warp_shuffle_16(
    mode: WarpShuffleMode,
    mask: u32,
    value: u16,
    b: u32,
    width: u32,
) -> (u16, bool) {
    let (value, oob) = unsafe { warp_shuffle_32(mode, mask, value as u32, b, width) };
    ((value as u16), oob)
}

unsafe fn warp_shuffle_8(
    mode: WarpShuffleMode,
    mask: u32,
    value: u8,
    b: u32,
    width: u32,
) -> (u8, bool) {
    let (value, oob) = unsafe { warp_shuffle_32(mode, mask, value as u32, b, width) };
    ((value as u8), oob)
}
