//! Mid-level intrinsics that take an ordering parameter and emulate specialized
//! instructions when not available (on lower compute capabilities).
//!
//! All functions are gpu-only, they do not work on the CPU.

// rustc thinks we don't use things because of gpu_only
#![allow(dead_code, unused_imports)]

use super::intrinsics;
use crate::gpu_only;
use core::sync::atomic::Ordering::{self, *};
use paste::paste;

#[gpu_only]
pub fn device_thread_fence(ordering: Ordering) {
    unsafe {
        if cfg!(target_feature = "compute_70") {
            if ordering == SeqCst {
                return intrinsics::fence_sc_device();
            }

            if ordering == Relaxed {
                return;
            }

            intrinsics::fence_acqrel_device();
        } else if ordering != Relaxed {
            intrinsics::membar_device();
        }
    }
}

#[gpu_only]
pub fn block_thread_fence(ordering: Ordering) {
    unsafe {
        if cfg!(target_feature = "compute_70") {
            if ordering == SeqCst {
                return intrinsics::fence_sc_block();
            }

            if ordering == Relaxed {
                return;
            }

            intrinsics::fence_acqrel_block();
        } else if ordering != Relaxed {
            intrinsics::membar_block();
        }
    }
}

#[gpu_only]
pub fn system_thread_fence(ordering: Ordering) {
    unsafe {
        if cfg!(target_feature = "compute_70") {
            if ordering == SeqCst {
                return intrinsics::fence_sc_system();
            }

            if ordering == Relaxed {
                return;
            }

            intrinsics::fence_acqrel_system();
        } else if ordering != Relaxed {
            intrinsics::membar_system();
        }
    }
}

macro_rules! load {
    ($($type:ty, $width:literal, $scope:ident),* $(,)?) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                pub unsafe fn [<atomic_load_ $width _ $scope>](ptr: *mut $type, ordering: Ordering) -> $type {
                    if cfg!(target_feature = "compute_70") {
                        match ordering {
                            SeqCst => {
                                intrinsics::[<fence_sc_ $scope>]();
                                intrinsics::[<atomic_load_acquire_ $width _ $scope>](ptr)
                            },
                            Acquire => {
                                intrinsics::[<atomic_load_acquire_ $width _ $scope>](ptr)
                            }
                            Relaxed => {
                                intrinsics::[<atomic_load_relaxed_ $width _ $scope>](ptr)
                            },
                            _ => panic!("Invalid Ordering for atomic load")
                        }
                    } else {
                        match ordering {
                            SeqCst => {
                                intrinsics::[<membar_ $scope>]();
                                let val = intrinsics::[<atomic_load_volatile_ $width _ $scope>](ptr);
                                intrinsics::[<membar_ $scope>]();
                                val
                            },
                            Acquire => {
                                let val = intrinsics::[<atomic_load_volatile_ $width _ $scope>](ptr);
                                intrinsics::[<membar_ $scope>]();
                                val
                            }
                            Relaxed => {
                                intrinsics::[<atomic_load_volatile_ $width _ $scope>](ptr)
                            },
                            _ => panic!("Invalid Ordering for atomic load")
                        }
                    }
                }
            }
        )*
    }
}

#[rustfmt::skip]
load!(
    u32, 32, device,
    u64, 64, device,
    u32, 32, block,
    u64, 64, block,
    u32, 32, system,
    u64, 64, system,
);

macro_rules! store {
    ($($type:ty, $width:literal, $scope:ident),* $(,)?) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                pub unsafe fn [<atomic_store_ $width _ $scope>](ptr: *mut $type, ordering: Ordering, val: $type) {
                    if cfg!(target_feature = "compute_70") {
                        match ordering {
                            SeqCst => {
                                intrinsics::[<fence_sc_ $scope>]();
                                intrinsics::[<atomic_store_relaxed_ $width _ $scope>](ptr, val);
                            },
                            Release => {
                                intrinsics::[<atomic_store_release_ $width _ $scope>](ptr, val);
                            }
                            Relaxed => {
                                intrinsics::[<atomic_store_relaxed_ $width _ $scope>](ptr, val);
                            },
                            _ => panic!("Invalid Ordering for atomic store")
                        }
                    } else {
                        match ordering {
                            SeqCst | Release => {
                                intrinsics::[<membar_ $scope>]();
                                intrinsics::[<atomic_store_volatile_ $width _ $scope>](ptr, val);
                            },
                            Relaxed => {
                                intrinsics::[<atomic_store_volatile_ $width _ $scope>](ptr, val);
                            },
                            _ => panic!("Invalid Ordering for atomic store")
                        }
                    }
                }
            }
        )*
    }
}

#[rustfmt::skip]
store!(
    u32, 32, device,
    u64, 64, device,
    u32, 32, block,
    u64, 64, block,
    u32, 32, system,
    u64, 64, system,
);

macro_rules! inner_fetch_ops_1_param {
    ($($type:ty, $op:ident, $scope:ident),* $(,)?) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                pub unsafe fn [<atomic_fetch_ $op _ $type _ $scope>](ptr: *mut $type, ordering: Ordering, val: $type) -> $type {
                    if cfg!(target_feature = "compute_70") {
                        match ordering {
                            SeqCst => {
                                intrinsics::[<fence_sc_ $scope>]();
                                intrinsics::[<atomic_fetch_ $op _acquire_ $type _ $scope>](ptr, val)
                            },
                            Acquire => intrinsics::[<atomic_fetch_ $op _acquire_ $type _ $scope>](ptr, val),
                            AcqRel => intrinsics::[<atomic_fetch_ $op _acqrel_ $type _ $scope>](ptr, val),
                            Release => intrinsics::[<atomic_fetch_ $op _release_ $type _ $scope>](ptr, val),
                            Relaxed => intrinsics::[<atomic_fetch_ $op _relaxed_ $type _ $scope>](ptr, val),
                            _ => unimplemented!("Weird ordering added by core")
                        }
                    } else {
                        match ordering {
                            SeqCst | AcqRel => {
                                intrinsics::[<membar_ $scope>]();
                                let val = intrinsics::[<atomic_fetch_ $op _volatile_ $type _ $scope>](ptr, val);
                                intrinsics::[<membar_ $scope>]();
                                val
                            },
                            Acquire => {
                                let val = intrinsics::[<atomic_fetch_ $op _volatile_ $type _ $scope>](ptr, val);
                                intrinsics::[<membar_ $scope>]();
                                val
                            },
                            Release => {
                                intrinsics::[<membar_ $scope>]();
                                intrinsics::[<atomic_fetch_ $op _volatile_ $type _ $scope>](ptr, val)
                            },
                            Relaxed => {
                                intrinsics::[<atomic_fetch_ $op _volatile_ $type _ $scope>](ptr, val)
                            },
                            _ => unimplemented!("Weird ordering added by core")
                        }
                    }
                }
            }
        )*
    }
}

macro_rules! fetch_ops_1_param {
    ($($op:ident => ($($type:ident),*)),* $(,)?) => {
        $(
            // every atomic function has a block, device, and system variant
            inner_fetch_ops_1_param!(
                $(
                    $type, $op, block,
                    $type, $op, device,
                    $type, $op, system,
                )*
            );
        )*
    };
}

fetch_ops_1_param! {
    and => (u32, u64, i32, i64, f32, f64),
    or => (u32, u64, i32, i64, f32, f64),
    xor => (u32, u64, i32, i64, f32, f64),
    add => (u32, u64, i32, i64, f32, f64),
    sub => (u32, u64, i32, i64, f32, f64),
    min => (u32, u64, i32, i64),
    max => (u32, u64, i32, i64),
    exch => (u32, u64, i32, i64, f32, f64),
}

macro_rules! inner_cas {
    ($($type:ty, $scope:ident),* $(,)?) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                pub unsafe fn [<atomic_compare_and_swap_ $type _ $scope>](ptr: *mut $type, current: $type, new: $type, ordering: Ordering) -> $type {
                    if cfg!(target_feature = "compute_70") {
                        match ordering {
                            SeqCst => {
                                intrinsics::[<fence_sc_ $scope>]();
                                intrinsics::[<atomic_fetch_cas_acquire_ $type _ $scope>](ptr, current, new)
                            },
                            Acquire => intrinsics::[<atomic_fetch_cas_acquire_ $type _ $scope>](ptr, current, new),
                            AcqRel => intrinsics::[<atomic_fetch_cas_acqrel_ $type _ $scope>](ptr, current, new),
                            Release => intrinsics::[<atomic_fetch_cas_release_ $type _ $scope>](ptr, current, new),
                            Relaxed => intrinsics::[<atomic_fetch_cas_relaxed_ $type _ $scope>](ptr, current, new),
                            _ => unimplemented!("Weird ordering added by core")
                        }
                    } else {
                        match ordering {
                            SeqCst | AcqRel => {
                                intrinsics::[<membar_ $scope>]();
                                let val = intrinsics::[<atomic_fetch_cas_volatile_ $type _ $scope>](ptr, current, new);
                                intrinsics::[<membar_ $scope>]();
                                val
                            },
                            Acquire => {
                                let val = intrinsics::[<atomic_fetch_cas_volatile_ $type _ $scope>](ptr, current, new);
                                intrinsics::[<membar_ $scope>]();
                                val
                            },
                            Release => {
                                intrinsics::[<membar_ $scope>]();
                                intrinsics::[<atomic_fetch_cas_volatile_ $type _ $scope>](ptr, current, new)
                            },
                            Relaxed => {
                                intrinsics::[<atomic_fetch_cas_volatile_ $type _ $scope>](ptr, current, new)
                            },
                            _ => unimplemented!("Weird ordering added by core")
                        }
                    }
                }
            }
        )*
    }
}

macro_rules! impl_cas {
    ($($type:ident),* $(,)?) => {
        $(
            inner_cas!(
                    $type, block,
                    $type, device,
                    $type, system,
            );
        )*
    };
}

impl_cas! {
    u32, u64, i32, i64, f32, f64
}
