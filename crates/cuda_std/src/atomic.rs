//! Atomic Types for modification of numbers in multiple threads in a sound way.
//!
//! # Core Interop
//!
//! Every type in this module works on the CPU (targets outside of nvptx). However, [`core::sync::atomic`] types
//! do **NOT** work on the GPU currently. This is because CUDA atomics have some fundamental differences
//! that make representing them fully with existing core types impossible:
//!
//! - CUDA has block-scoped, device-scoped, and system-scoped atomics, core does not make such a distinction (obviously).
//! - CUDA trivially supports relaxed/acquire/release orderings on most architectures, but SeqCst and other orderings use
//! specialized instructions on compute capabilities 7.x+, but can be emulated with fences/membars on 7.x >. This makes it difficult
//! to hide away such details in the codegen.
//! - CUDA has hardware atomic floats, core does not.
//! - CUDA makes the distinction between "fetch, do operation, read" (`atom`) and "do operation" (`red`).
//! - Core thinks CUDA supports 8 and 16 bit atomics, this is a bug in the nvptx target but it is nevertheless an annoying detail
//! to silently trap on.
//!
//! Therefore we chose to go with the approach of implementing all atomics inside cuda_std. In the future, we may support
//! a subset of core atomics, but for now, you will have to use cuda_std atomics.

#![allow(unused_unsafe, warnings)]

pub mod intrinsics;
pub mod mid;

use core::cell::UnsafeCell;
use core::sync::atomic::Ordering;
use paste::paste;

#[allow(dead_code)]
fn fail_order(order: Ordering) -> Ordering {
    match order {
        Ordering::Release | Ordering::Relaxed => Ordering::Relaxed,
        Ordering::Acquire | Ordering::AcqRel => Ordering::Acquire,
        Ordering::SeqCst => Ordering::SeqCst,
        x => x,
    }
}

macro_rules! scope_doc {
    (device) => {
        "a single device (GPU)."
    };
    (block) => {
        "a single thread block (also called a CTA, cooperative thread array)."
    };
    (system) => {
        "the entire system."
    };
}

macro_rules! safety_doc {
    ($($unsafety:ident)?) => {
        $(
            concat!(
                "# Safety\n",
                concat!("This function is ", stringify!($unsafety), " because it does not synchronize\n"),
                "across the entire GPU or System, which leaves it open for data races if used incorrectly"
            )
        )?
    };
}

// taken from stdlib compare_and_swap docs
fn double_ordering_from_one(ordering: Ordering) -> (Ordering, Ordering) {
    match ordering {
        Ordering::Relaxed => (Ordering::Relaxed, Ordering::Relaxed),
        Ordering::Acquire => (Ordering::Acquire, Ordering::Acquire),
        Ordering::Release => (Ordering::Release, Ordering::Relaxed),
        Ordering::AcqRel => (Ordering::AcqRel, Ordering::Acquire),
        Ordering::SeqCst => (Ordering::SeqCst, Ordering::SeqCst),
        _ => unreachable!(),
    }
}

macro_rules! atomic_float {
    ($float_ty:ident, $atomic_ty:ident, $align:tt, $scope:ident, $width:tt $(,$unsafety:ident)?) => {
        #[doc = concat!("A ", stringify!($width), "-bit float type which can be safely shared between threads and synchronizes across ", scope_doc!($scope))]
        ///
        /// This type is guaranteed to have the same memory representation as the underlying integer
        /// type [`
        #[doc = stringify!($float_ty)]
        /// `].
        ///
        /// The functions on this type map to hardware instructions on CUDA platforms, and are emulated
        /// with a CAS loop on the CPU (non-CUDA targets).
        #[repr(C, align($align))]
        pub struct $atomic_ty {
            v: UnsafeCell<$float_ty>,
        }

        // SAFETY: atomic ops make sure this is fine
        unsafe impl Sync for $atomic_ty {}

        impl $atomic_ty {
            paste! {
                /// Creates a new atomic float.
                pub const fn new(v: $float_ty) -> $atomic_ty {
                    Self {
                        v: UnsafeCell::new(v),
                    }
                }

                /// Consumes the atomic and returns the contained value.
                pub fn into_inner(self) -> $float_ty {
                    self.v.into_inner()
                }

                #[cfg(not(target_os = "cuda"))]
                fn as_atomic_bits(&self) -> &core::sync::atomic::[<AtomicU $width>] {
                    // SAFETY: AtomicU32/U64 pointers are compatible with UnsafeCell<u32/u64>.
                    unsafe {
                        core::mem::transmute(self)
                    }
                }

                #[cfg(not(target_os = "cuda"))]
                fn update_with(&self, order: Ordering, mut func: impl FnMut($float_ty) -> $float_ty) -> $float_ty {
                    let res = self
                        .as_atomic_bits()
                        .fetch_update(order, fail_order(order), |prev| {
                            Some(func($float_ty::from_bits(prev))).map($float_ty::to_bits)
                        }).unwrap();
                    $float_ty::from_bits(res)
                }

                /// Adds to the current value, returning the previous value **before** the addition.
                ///
                $(#[doc = safety_doc!($unsafety)])?
                pub $($unsafety)? fn fetch_add(&self, val: $float_ty, order: Ordering) -> $float_ty {
                    #[cfg(target_os = "cuda")]
                    // SAFETY: data races are prevented by atomic intrinsics and the pointer we get is valid.
                    unsafe {
                        mid::[<atomic_fetch_add_ $float_ty _ $scope>](self.v.get(), order, val)
                    }
                    #[cfg(not(target_os = "cuda"))]
                    self.update_with(order, |v| v + val)
                }

                /// Subtracts from the current value, returning the previous value **before** the subtraction.
                ///
                /// Note, this is actually implemented as `old + (-new)`, CUDA does not have a specialized sub instruction.
                ///
                $(#[doc = safety_doc!($unsafety)])?
                pub $($unsafety)? fn fetch_sub(&self, val: $float_ty, order: Ordering) -> $float_ty {
                    #[cfg(target_os = "cuda")]
                    // SAFETY: data races are prevented by atomic intrinsics and the pointer we get is valid.
                    unsafe {
                        mid::[<atomic_fetch_sub_ $float_ty _ $scope>](self.v.get(), order, val)
                    }
                    #[cfg(not(target_os = "cuda"))]
                    self.update_with(order, |v| v - val)
                }

                /// Bitwise "and" with the current value. Returns the value **before** the "and".
                ///
                $(#[doc = safety_doc!($unsafety)])?
                pub $($unsafety)? fn fetch_and(&self, val: $float_ty, order: Ordering) -> $float_ty {
                    #[cfg(target_os = "cuda")]
                    // SAFETY: data races are prevented by atomic intrinsics and the pointer we get is valid.
                    unsafe {
                        mid::[<atomic_fetch_and_ $float_ty _ $scope>](self.v.get(), order, val)
                    }
                    #[cfg(not(target_os = "cuda"))]
                    self.update_with(order, |v| $float_ty::from_bits(v.to_bits() & val.to_bits()))
                }

                /// Bitwise "or" with the current value. Returns the value **before** the "or".
                ///
                $(#[doc = safety_doc!($unsafety)])?
                pub $($unsafety)? fn fetch_or(&self, val: $float_ty, order: Ordering) -> $float_ty {
                    #[cfg(target_os = "cuda")]
                    // SAFETY: data races are prevented by atomic intrinsics and the pointer we get is valid.
                    unsafe {
                        mid::[<atomic_fetch_or_ $float_ty _ $scope>](self.v.get(), order, val)
                    }
                    #[cfg(not(target_os = "cuda"))]
                    self.update_with(order, |v| $float_ty::from_bits(v.to_bits() | val.to_bits()))
                }

                /// Bitwise "xor" with the current value. Returns the value **before** the "xor".
                ///
                $(#[doc = safety_doc!($unsafety)])?
                pub $($unsafety)? fn fetch_xor(&self, val: $float_ty, order: Ordering) -> $float_ty {
                    #[cfg(target_os = "cuda")]
                    // SAFETY: data races are prevented by atomic intrinsics and the pointer we get is valid.
                    unsafe {
                        mid::[<atomic_fetch_xor_ $float_ty _ $scope>](self.v.get(), order, val)
                    }
                    #[cfg(not(target_os = "cuda"))]
                    self.update_with(order, |v| $float_ty::from_bits(v.to_bits() ^ val.to_bits()))
                }

                /// Atomically loads the value behind this atomic.
                ///
                /// `load` takes an [`Ordering`] argument which describes the memory ordering of this operation.
                /// Possible values are [`Ordering::SeqCst`], [`Ordering::Acquire`], and [`Ordering::Relaxed`].
                ///
                /// # Panics
                ///
                /// Panics if `order` is [`Ordering::Release`] or [`Ordering::AcqRel`].
                ///
                $(#[doc = safety_doc!($unsafety)])?
                pub $($unsafety)? fn load(&self, order: Ordering) -> $float_ty {
                    #[cfg(target_os = "cuda")]
                    unsafe {
                        let val = mid::[<atomic_load_ $width _ $scope>](self.v.get().cast(), order);
                        $float_ty::from_bits(val)
                    }
                    #[cfg(not(target_os = "cuda"))]
                    {
                        let val = self.as_atomic_bits().load(order);
                        $float_ty::from_bits(val)
                    }
                }

                /// Atomically stores a value into this atomic.
                ///
                /// `store` takes an [`Ordering`] argument which describes the memory ordering of this operation.
                /// Possible values are [`Ordering::SeqCst`], [`Ordering::Release`], and [`Ordering::Relaxed`].
                ///
                /// # Panics
                ///
                /// Panics if `order` is [`Ordering::Acquire`] or [`Ordering::AcqRel`].
                ///
                $(#[doc = safety_doc!($unsafety)])?
                pub $($unsafety)? fn store(&self, val: $float_ty, order: Ordering) {
                    #[cfg(target_os = "cuda")]
                    unsafe {
                        mid::[<atomic_store_ $width _ $scope>](self.v.get().cast(), order, val.to_bits());
                    }
                    #[cfg(not(target_os = "cuda"))]
                    self.as_atomic_bits().store(val.to_bits(), order);
                }

                // $(#[doc = safety_doc!($unsafety)])?
                // pub $($unsafety)? fn compare_and_swap(&self, current: f32, new: f32, order: Ordering) -> Result<$float_ty, $float_ty> {
                //     #[cfg(target_os = "cuda")]
                //     unsafe {
                //         let res = mid::[<atomic_compare_and_swap_ $float_ty _ $scope>](self.v.get().cast(), order, current, new);
                //     }

                //     #[cfg(not(target_os = "cuda"))]
                //     self.as_atomic_bits().compare_exchange
                // }
            }
        }
    };
}

atomic_float!(f32, AtomicF32, 4, device, 32);
atomic_float!(f64, AtomicF64, 8, device, 64);
atomic_float!(f32, BlockAtomicF32, 4, block, 32, unsafe);
atomic_float!(f64, BlockAtomicF64, 8, block, 64, unsafe);
atomic_float!(f32, SystemAtomicF32, 4, system, 32);
atomic_float!(f64, SystemAtomicF64, 8, system, 64);
