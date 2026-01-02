//! Raw CUDA-specific atomic functions that map to PTX instructions.

use crate::gpu_only;
use core::arch::asm;
use core::concat;
use paste::paste;

#[gpu_only]
pub unsafe fn membar_device() {
    asm!("membar.gl;");
}

#[gpu_only]
pub unsafe fn membar_block() {
    asm!("membar.cta;");
}

#[gpu_only]
pub unsafe fn membar_system() {
    asm!("membar.sys;");
}

#[gpu_only]
pub unsafe fn fence_sc_device() {
    asm!("fence.sc.gl;");
}

#[gpu_only]
pub unsafe fn fence_sc_block() {
    asm!("fence.sc.cta;");
}

#[gpu_only]
pub unsafe fn fence_sc_system() {
    asm!("fence.sc.sys;");
}

#[gpu_only]
pub unsafe fn fence_acqrel_device() {
    asm!("fence.acq_rel.gl;");
}

#[gpu_only]
pub unsafe fn fence_acqrel_block() {
    asm!("fence.acq_rel.sys;");
}

#[gpu_only]
pub unsafe fn fence_acqrel_system() {
    asm!("fence.acq_rel.sys;");
}

#[allow(unused_macros)]
macro_rules! load_scope {
    (volatile, $scope:ident) => {
        ""
    };
    ($ordering:ident, $scope:ident) => {
        concat!(".", stringify!($scope))
    };
}

macro_rules! load {
    ($($ordering:ident, $width:literal, $scope:ident, $scope_asm:ident),* $(,)*) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                #[doc = concat!("Performs a ", stringify!($ordering), " atomic load at the ", stringify!($scope), " level with a width of ", stringify!($width), " bits")]
                pub unsafe fn [<atomic_load_ $ordering _ $width _ $scope>](ptr: *const [<u $width>]) -> [<u $width>] {
                    let mut out;
                    asm!(
                        concat!("ld.", stringify!($ordering), load_scope!($ordering, $scope), ".", stringify!([<u $width>]), " {}, [{}];"),
                        out([<reg $width>]) out,
                        in(reg64) ptr
                    );
                    out
                }
            }
        )*
    };
}

load! {
    relaxed, 32, device, gpu,
    acquire, 32, device, gpu,
    volatile, 32, device, gpu,

    relaxed, 64, device, gpu,
    acquire, 64, device, gpu,
    volatile, 64, device, gpu,

    relaxed, 32, block, cta,
    acquire, 32, block, cta,
    volatile, 32, block, cta,

    relaxed, 64, block, cta,
    acquire, 64, block, cta,
    volatile, 64, block, cta,

    relaxed, 32, system, sys,
    acquire, 32, system, sys,
    volatile, 32, system, sys,

    relaxed, 64, system, sys,
    acquire, 64, system, sys,
    volatile, 64, system, sys,
}

macro_rules! store {
    ($($ordering:ident, $width:literal, $scope:ident, $scope_asm:ident),* $(,)*) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                #[doc = concat!("Performs a ", stringify!($ordering), " atomic store at the ", stringify!($scope), " level with a width of ", stringify!($width), " bits")]
                pub unsafe fn [<atomic_store_ $ordering _ $width _ $scope>](ptr: *mut [<u $width>], val: [<u $width>]) {
                    asm!(
                        concat!("st.", stringify!($ordering), load_scope!($ordering, $scope), ".", stringify!([<u $width>]), " [{}], {};"),
                        in(reg64) ptr,
                        in([<reg $width>]) val,
                    );
                }
            }
        )*
    };
}

store! {
    relaxed, 32, device, gpu,
    release, 32, device, gpu,
    volatile, 32, device, gpu,

    relaxed, 64, device, gpu,
    release, 64, device, gpu,
    volatile, 64, device, gpu,

    relaxed, 32, block, cta,
    release, 32, block, cta,
    volatile, 32, block, cta,

    relaxed, 64, block, cta,
    release, 64, block, cta,
    volatile, 64, block, cta,

    relaxed, 32, system, sys,
    release, 32, system, sys,
    volatile, 32, system, sys,

    relaxed, 64, system, sys,
    release, 64, system, sys,
    volatile, 64, system, sys,
}

#[allow(unused_macros)]
macro_rules! ptx_type {
    (i32) => {
        "s32"
    };
    (i64) => {
        "s64"
    };
    ($ty:ident) => {
        stringify!($ty)
    };
}

#[allow(unused_macros)]
macro_rules! ptx_type_for_logic_op {
    (u32) => {
        "b32"
    };
    (u64) => {
        "b64"
    };
    (i32) => {
        "b32"
    };
    (i64) => {
        "b64"
    };
    (f32) => {
        "b32"
    };
    (f64) => {
        "b64"
    };
}

#[allow(unused_macros)]
macro_rules! ordering {
    (volatile) => {
        ""
    };
    ($ordering:ident) => {
        concat!(stringify!($ordering), ".")
    };
}

macro_rules! atomic_fetch_op_2_reg {
    ($($ordering:ident, $op:ident, $width:literal, $type:ty, $scope:ident, $scope_asm:ident),* $(,)*) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                #[doc = concat!(
                    "Fetches the value in ptr, performs a ",
                    stringify!($op),
                    ", and returns the original value"
                )]
                pub unsafe fn [<atomic_fetch_ $op _ $ordering _ $type _ $scope>](ptr: *mut $type) -> $type {
                    let mut out;
                    asm!(
                        concat!(
                            "atom.",
                            ordering!($ordering),
                            stringify!($scope_asm),
                            ".",
                            stringify!($op),
                            ".",
                            ptx_type!($type),
                            " {}, [{}];"
                        ),
                        out([<reg $width>]) out,
                        in(reg64) ptr,
                    );
                    out
                }
            }
        )*
    };
}

atomic_fetch_op_2_reg! {
    // inc (unsigned)

    relaxed, inc, 32, u32, device, gpu,
    acquire, inc, 32, u32, device, gpu,
    release, inc, 32, u32, device, gpu,
    acqrel,  inc, 32, u32, device, gpu,
    volatile, inc, 32, u32, device, gpu,

    relaxed, inc, 64, u64, device, gpu,
    acquire, inc, 64, u64, device, gpu,
    release, inc, 64, u64, device, gpu,
    acqrel,  inc, 64, u64, device, gpu,
    volatile, inc, 64, u64, device, gpu,

    relaxed, inc, 32, u32, block, cta,
    acquire, inc, 32, u32, block, cta,
    release, inc, 32, u32, block, cta,
    acqrel,  inc, 32, u32, block, cta,
    volatile, inc, 32, u32, block, cta,

    relaxed, inc, 64, u64, block, cta,
    acquire, inc, 64, u64, block, cta,
    release, inc, 64, u64, block, cta,
    acqrel,  inc, 64, u64, block, cta,
    volatile, inc, 64, u64, block, cta,

    relaxed, inc, 32, u32, system, sys,
    acquire, inc, 32, u32, system, sys,
    release, inc, 32, u32, system, sys,
    acqrel,  inc, 32, u32, system, sys,
    volatile, inc, 32, u32, system, sys,

    relaxed, inc, 64, u64, system, sys,
    acquire, inc, 64, u64, system, sys,
    release, inc, 64, u64, system, sys,
    acqrel,  inc, 64, u64, system, sys,
    volatile, inc, 64, u64, system, sys,

    // inc (signed)

    relaxed, inc, 32, i32, device, gpu,
    acquire, inc, 32, i32, device, gpu,
    release, inc, 32, i32, device, gpu,
    acqrel,  inc, 32, i32, device, gpu,
    volatile, inc, 32, i32, device, gpu,

    relaxed, inc, 64, i64, device, gpu,
    acquire, inc, 64, i64, device, gpu,
    release, inc, 64, i64, device, gpu,
    acqrel,  inc, 64, i64, device, gpu,
    volatile, inc, 64, i64, device, gpu,

    relaxed, inc, 32, i32, block, cta,
    acquire, inc, 32, i32, block, cta,
    release, inc, 32, i32, block, cta,
    acqrel,  inc, 32, i32, block, cta,
    volatile, inc, 32, i32, block, cta,

    relaxed, inc, 64, i64, block, cta,
    acquire, inc, 64, i64, block, cta,
    release, inc, 64, i64, block, cta,
    acqrel,  inc, 64, i64, block, cta,
    volatile, inc, 64, i64, block, cta,

    relaxed, inc, 32, i32, system, sys,
    acquire, inc, 32, i32, system, sys,
    release, inc, 32, i32, system, sys,
    acqrel,  inc, 32, i32, system, sys,
    volatile, inc, 32, i32, system, sys,

    relaxed, inc, 64, i64, system, sys,
    acquire, inc, 64, i64, system, sys,
    release, inc, 64, i64, system, sys,
    acqrel,  inc, 64, i64, system, sys,
    volatile, inc, 64, i64, system, sys,

    // dec (unsigned)

    relaxed, dec, 32, u32, device, gpu,
    acquire, dec, 32, u32, device, gpu,
    release, dec, 32, u32, device, gpu,
    acqrel,  dec, 32, u32, device, gpu,
    volatile, dec, 32, u32, device, gpu,

    relaxed, dec, 64, u64, device, gpu,
    acquire, dec, 64, u64, device, gpu,
    release, dec, 64, u64, device, gpu,
    acqrel,  dec, 64, u64, device, gpu,
    volatile, dec, 64, u64, device, gpu,

    relaxed, dec, 32, u32, block, cta,
    acquire, dec, 32, u32, block, cta,
    release, dec, 32, u32, block, cta,
    acqrel,  dec, 32, u32, block, cta,
    volatile, dec, 32, u32, block, cta,

    relaxed, dec, 64, u64, block, cta,
    acquire, dec, 64, u64, block, cta,
    release, dec, 64, u64, block, cta,
    acqrel,  dec, 64, u64, block, cta,
    volatile, dec, 64, u64, block, cta,

    relaxed, dec, 32, u32, system, sys,
    acquire, dec, 32, u32, system, sys,
    release, dec, 32, u32, system, sys,
    acqrel,  dec, 32, u32, system, sys,
    volatile, dec, 32, u32, system, sys,

    relaxed, dec, 64, u64, system, sys,
    acquire, dec, 64, u64, system, sys,
    release, dec, 64, u64, system, sys,
    acqrel,  dec, 64, u64, system, sys,
    volatile, dec, 64, u64, system, sys,

    // dec (signed)

    relaxed, dec, 32, i32, device, gpu,
    acquire, dec, 32, i32, device, gpu,
    release, dec, 32, i32, device, gpu,
    acqrel,  dec, 32, i32, device, gpu,
    volatile, dec, 32, i32, device, gpu,

    relaxed, dec, 64, i64, device, gpu,
    acquire, dec, 64, i64, device, gpu,
    release, dec, 64, i64, device, gpu,
    acqrel,  dec, 64, i64, device, gpu,
    volatile, dec, 64, i64, device, gpu,

    relaxed, dec, 32, i32, block, cta,
    acquire, dec, 32, i32, block, cta,
    release, dec, 32, i32, block, cta,
    acqrel,  dec, 32, i32, block, cta,
    volatile, dec, 32, i32, block, cta,

    relaxed, dec, 64, i64, block, cta,
    acquire, dec, 64, i64, block, cta,
    release, dec, 64, i64, block, cta,
    acqrel,  dec, 64, i64, block, cta,
    volatile, dec, 64, i64, block, cta,

    relaxed, dec, 32, i32, system, sys,
    acquire, dec, 32, i32, system, sys,
    release, dec, 32, i32, system, sys,
    acqrel,  dec, 32, i32, system, sys,
    volatile, dec, 32, i32, system, sys,

    relaxed, dec, 64, i64, system, sys,
    acquire, dec, 64, i64, system, sys,
    release, dec, 64, i64, system, sys,
    acqrel,  dec, 64, i64, system, sys,
    volatile, dec, 64, i64, system, sys,
}

macro_rules! atomic_fetch_logic_op_3_reg {
    ($($ordering:ident, $op:ident, $width:literal, $type:ty, $scope:ident, $scope_asm:ident),* $(,)*) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                #[doc = concat!(
                    "Fetches the value in ptr, performs a ",
                    stringify!($op),
                    ", and returns the original value"
                )]
                pub unsafe fn [<atomic_fetch_ $op _ $ordering _ $type _ $scope>](ptr: *mut $type, val: $type) -> $type {
                    let mut out;
                    asm!(
                        concat!(
                            "atom.",
                            ordering!($ordering),
                            stringify!($scope_asm),
                            ".",
                            stringify!($op),
                            ".",
                            ptx_type_for_logic_op!($type),
                            " {}, [{}], {};"
                        ),
                        out([<reg $width>]) out,
                        in(reg64) ptr,
                        in([<reg $width>]) val,
                    );
                    out
                }
            }
        )*
    };
}

atomic_fetch_logic_op_3_reg! {
    // and

    relaxed, and, 32, u32, device, gpu,
    acquire, and, 32, u32, device, gpu,
    release, and, 32, u32, device, gpu,
    acqrel,  and, 32, u32, device, gpu,
    volatile, and, 32, u32, device, gpu,

    relaxed, and, 64, u64, device, gpu,
    acquire, and, 64, u64, device, gpu,
    release, and, 64, u64, device, gpu,
    acqrel,  and, 64, u64, device, gpu,
    volatile, and, 64, u64, device, gpu,

    relaxed, and, 32, u32, block, cta,
    acquire, and, 32, u32, block, cta,
    release, and, 32, u32, block, cta,
    acqrel,  and, 32, u32, block, cta,
    volatile, and, 32, u32, block, cta,

    relaxed, and, 64, u64, block, cta,
    acquire, and, 64, u64, block, cta,
    release, and, 64, u64, block, cta,
    acqrel,  and, 64, u64, block, cta,
    volatile, and, 64, u64, block, cta,

    relaxed, and, 32, u32, system, sys,
    acquire, and, 32, u32, system, sys,
    release, and, 32, u32, system, sys,
    acqrel,  and, 32, u32, system, sys,
    volatile, and, 32, u32, system, sys,

    relaxed, and, 64, u64, system, sys,
    acquire, and, 64, u64, system, sys,
    release, and, 64, u64, system, sys,
    acqrel,  and, 64, u64, system, sys,
    volatile, and, 64, u64, system, sys,

    relaxed, and, 32, i32, device, gpu,
    acquire, and, 32, i32, device, gpu,
    release, and, 32, i32, device, gpu,
    acqrel,  and, 32, i32, device, gpu,
    volatile, and, 32, i32, device, gpu,

    relaxed, and, 64, i64, device, gpu,
    acquire, and, 64, i64, device, gpu,
    release, and, 64, i64, device, gpu,
    acqrel,  and, 64, i64, device, gpu,
    volatile, and, 64, i64, device, gpu,

    relaxed, and, 32, i32, block, cta,
    acquire, and, 32, i32, block, cta,
    release, and, 32, i32, block, cta,
    acqrel,  and, 32, i32, block, cta,
    volatile, and, 32, i32, block, cta,

    relaxed, and, 64, i64, block, cta,
    acquire, and, 64, i64, block, cta,
    release, and, 64, i64, block, cta,
    acqrel,  and, 64, i64, block, cta,
    volatile, and, 64, i64, block, cta,

    relaxed, and, 32, i32, system, sys,
    acquire, and, 32, i32, system, sys,
    release, and, 32, i32, system, sys,
    acqrel,  and, 32, i32, system, sys,
    volatile, and, 32, i32, system, sys,

    relaxed, and, 64, i64, system, sys,
    acquire, and, 64, i64, system, sys,
    release, and, 64, i64, system, sys,
    acqrel,  and, 64, i64, system, sys,
    volatile, and, 64, i64, system, sys,

    relaxed, and, 32, f32, device, gpu,
    acquire, and, 32, f32, device, gpu,
    release, and, 32, f32, device, gpu,
    acqrel,  and, 32, f32, device, gpu,
    volatile, and, 32, f32, device, gpu,

    relaxed, and, 64, f64, device, gpu,
    acquire, and, 64, f64, device, gpu,
    release, and, 64, f64, device, gpu,
    acqrel,  and, 64, f64, device, gpu,
    volatile, and, 64, f64, device, gpu,

    relaxed, and, 32, f32, block, cta,
    acquire, and, 32, f32, block, cta,
    release, and, 32, f32, block, cta,
    acqrel,  and, 32, f32, block, cta,
    volatile, and, 32, f32, block, cta,

    relaxed, and, 64, f64, block, cta,
    acquire, and, 64, f64, block, cta,
    release, and, 64, f64, block, cta,
    acqrel,  and, 64, f64, block, cta,
    volatile, and, 64, f64, block, cta,

    relaxed, and, 32, f32, system, sys,
    acquire, and, 32, f32, system, sys,
    release, and, 32, f32, system, sys,
    acqrel,  and, 32, f32, system, sys,
    volatile, and, 32, f32, system, sys,

    relaxed, and, 64, f64, system, sys,
    acquire, and, 64, f64, system, sys,
    release, and, 64, f64, system, sys,
    acqrel,  and, 64, f64, system, sys,
    volatile, and, 64, f64, system, sys,

    // or

    relaxed, or, 32, u32, device, gpu,
    acquire, or, 32, u32, device, gpu,
    release, or, 32, u32, device, gpu,
    acqrel,  or, 32, u32, device, gpu,
    volatile, or, 32, u32, device, gpu,

    relaxed, or, 64, u64, device, gpu,
    acquire, or, 64, u64, device, gpu,
    release, or, 64, u64, device, gpu,
    acqrel,  or, 64, u64, device, gpu,
    volatile, or, 64, u64, device, gpu,

    relaxed, or, 32, u32, block, cta,
    acquire, or, 32, u32, block, cta,
    release, or, 32, u32, block, cta,
    acqrel,  or, 32, u32, block, cta,
    volatile, or, 32, u32, block, cta,

    relaxed, or, 64, u64, block, cta,
    acquire, or, 64, u64, block, cta,
    release, or, 64, u64, block, cta,
    acqrel,  or, 64, u64, block, cta,
    volatile, or, 64, u64, block, cta,

    relaxed, or, 32, u32, system, sys,
    acquire, or, 32, u32, system, sys,
    release, or, 32, u32, system, sys,
    acqrel,  or, 32, u32, system, sys,
    volatile, or, 32, u32, system, sys,

    relaxed, or, 64, u64, system, sys,
    acquire, or, 64, u64, system, sys,
    release, or, 64, u64, system, sys,
    acqrel,  or, 64, u64, system, sys,
    volatile, or, 64, u64, system, sys,

    relaxed, or, 32, i32, device, gpu,
    acquire, or, 32, i32, device, gpu,
    release, or, 32, i32, device, gpu,
    acqrel,  or, 32, i32, device, gpu,
    volatile, or, 32, i32, device, gpu,

    relaxed, or, 64, i64, device, gpu,
    acquire, or, 64, i64, device, gpu,
    release, or, 64, i64, device, gpu,
    acqrel,  or, 64, i64, device, gpu,
    volatile, or, 64, i64, device, gpu,

    relaxed, or, 32, i32, block, cta,
    acquire, or, 32, i32, block, cta,
    release, or, 32, i32, block, cta,
    acqrel,  or, 32, i32, block, cta,
    volatile, or, 32, i32, block, cta,

    relaxed, or, 64, i64, block, cta,
    acquire, or, 64, i64, block, cta,
    release, or, 64, i64, block, cta,
    acqrel,  or, 64, i64, block, cta,
    volatile, or, 64, i64, block, cta,

    relaxed, or, 32, i32, system, sys,
    acquire, or, 32, i32, system, sys,
    release, or, 32, i32, system, sys,
    acqrel,  or, 32, i32, system, sys,
    volatile, or, 32, i32, system, sys,

    relaxed, or, 64, i64, system, sys,
    acquire, or, 64, i64, system, sys,
    release, or, 64, i64, system, sys,
    acqrel,  or, 64, i64, system, sys,
    volatile, or, 64, i64, system, sys,

    relaxed, or, 32, f32, device, gpu,
    acquire, or, 32, f32, device, gpu,
    release, or, 32, f32, device, gpu,
    acqrel,  or, 32, f32, device, gpu,
    volatile, or, 32, f32, device, gpu,

    relaxed, or, 64, f64, device, gpu,
    acquire, or, 64, f64, device, gpu,
    release, or, 64, f64, device, gpu,
    acqrel,  or, 64, f64, device, gpu,
    volatile, or, 64, f64, device, gpu,

    relaxed, or, 32, f32, block, cta,
    acquire, or, 32, f32, block, cta,
    release, or, 32, f32, block, cta,
    acqrel,  or, 32, f32, block, cta,
    volatile, or, 32, f32, block, cta,

    relaxed, or, 64, f64, block, cta,
    acquire, or, 64, f64, block, cta,
    release, or, 64, f64, block, cta,
    acqrel,  or, 64, f64, block, cta,
    volatile, or, 64, f64, block, cta,

    relaxed, or, 32, f32, system, sys,
    acquire, or, 32, f32, system, sys,
    release, or, 32, f32, system, sys,
    acqrel,  or, 32, f32, system, sys,
    volatile, or, 32, f32, system, sys,

    relaxed, or, 64, f64, system, sys,
    acquire, or, 64, f64, system, sys,
    release, or, 64, f64, system, sys,
    acqrel,  or, 64, f64, system, sys,
    volatile, or, 64, f64, system, sys,

    // xor

    relaxed, xor, 32, u32, device, gpu,
    acquire, xor, 32, u32, device, gpu,
    release, xor, 32, u32, device, gpu,
    acqrel,  xor, 32, u32, device, gpu,
    volatile, xor, 32, u32, device, gpu,

    relaxed, xor, 64, u64, device, gpu,
    acquire, xor, 64, u64, device, gpu,
    release, xor, 64, u64, device, gpu,
    acqrel,  xor, 64, u64, device, gpu,
    volatile, xor, 64, u64, device, gpu,

    relaxed, xor, 32, u32, block, cta,
    acquire, xor, 32, u32, block, cta,
    release, xor, 32, u32, block, cta,
    acqrel,  xor, 32, u32, block, cta,
    volatile, xor, 32, u32, block, cta,

    relaxed, xor, 64, u64, block, cta,
    acquire, xor, 64, u64, block, cta,
    release, xor, 64, u64, block, cta,
    acqrel,  xor, 64, u64, block, cta,
    volatile, xor, 64, u64, block, cta,

    relaxed, xor, 32, u32, system, sys,
    acquire, xor, 32, u32, system, sys,
    release, xor, 32, u32, system, sys,
    acqrel,  xor, 32, u32, system, sys,
    volatile, xor, 32, u32, system, sys,

    relaxed, xor, 64, u64, system, sys,
    acquire, xor, 64, u64, system, sys,
    release, xor, 64, u64, system, sys,
    acqrel,  xor, 64, u64, system, sys,
    volatile, xor, 64, u64, system, sys,

    relaxed, xor, 32, i32, device, gpu,
    acquire, xor, 32, i32, device, gpu,
    release, xor, 32, i32, device, gpu,
    acqrel,  xor, 32, i32, device, gpu,
    volatile, xor, 32, i32, device, gpu,

    relaxed, xor, 64, i64, device, gpu,
    acquire, xor, 64, i64, device, gpu,
    release, xor, 64, i64, device, gpu,
    acqrel,  xor, 64, i64, device, gpu,
    volatile, xor, 64, i64, device, gpu,

    relaxed, xor, 32, i32, block, cta,
    acquire, xor, 32, i32, block, cta,
    release, xor, 32, i32, block, cta,
    acqrel,  xor, 32, i32, block, cta,
    volatile, xor, 32, i32, block, cta,

    relaxed, xor, 64, i64, block, cta,
    acquire, xor, 64, i64, block, cta,
    release, xor, 64, i64, block, cta,
    acqrel,  xor, 64, i64, block, cta,
    volatile, xor, 64, i64, block, cta,

    relaxed, xor, 32, i32, system, sys,
    acquire, xor, 32, i32, system, sys,
    release, xor, 32, i32, system, sys,
    acqrel,  xor, 32, i32, system, sys,
    volatile, xor, 32, i32, system, sys,

    relaxed, xor, 64, i64, system, sys,
    acquire, xor, 64, i64, system, sys,
    release, xor, 64, i64, system, sys,
    acqrel,  xor, 64, i64, system, sys,
    volatile, xor, 64, i64, system, sys,

    relaxed, xor, 32, f32, device, gpu,
    acquire, xor, 32, f32, device, gpu,
    release, xor, 32, f32, device, gpu,
    acqrel,  xor, 32, f32, device, gpu,
    volatile, xor, 32, f32, device, gpu,

    relaxed, xor, 64, f64, device, gpu,
    acquire, xor, 64, f64, device, gpu,
    release, xor, 64, f64, device, gpu,
    acqrel,  xor, 64, f64, device, gpu,
    volatile, xor, 64, f64, device, gpu,

    relaxed, xor, 32, f32, block, cta,
    acquire, xor, 32, f32, block, cta,
    release, xor, 32, f32, block, cta,
    acqrel,  xor, 32, f32, block, cta,
    volatile, xor, 32, f32, block, cta,

    relaxed, xor, 64, f64, block, cta,
    acquire, xor, 64, f64, block, cta,
    release, xor, 64, f64, block, cta,
    acqrel,  xor, 64, f64, block, cta,
    volatile, xor, 64, f64, block, cta,

    relaxed, xor, 32, f32, system, sys,
    acquire, xor, 32, f32, system, sys,
    release, xor, 32, f32, system, sys,
    acqrel,  xor, 32, f32, system, sys,
    volatile, xor, 32, f32, system, sys,

    relaxed, xor, 64, f64, system, sys,
    acquire, xor, 64, f64, system, sys,
    release, xor, 64, f64, system, sys,
    acqrel,  xor, 64, f64, system, sys,
    volatile, xor, 64, f64, system, sys,

    // exchange

    relaxed, exch, 32, u32, device, gpu,
    acquire, exch, 32, u32, device, gpu,
    release, exch, 32, u32, device, gpu,
    acqrel,  exch, 32, u32, device, gpu,
    volatile, exch, 32, u32, device, gpu,

    relaxed, exch, 64, u64, device, gpu,
    acquire, exch, 64, u64, device, gpu,
    release, exch, 64, u64, device, gpu,
    acqrel,  exch, 64, u64, device, gpu,
    volatile, exch, 64, u64, device, gpu,

    relaxed, exch, 32, u32, block, cta,
    acquire, exch, 32, u32, block, cta,
    release, exch, 32, u32, block, cta,
    acqrel,  exch, 32, u32, block, cta,
    volatile, exch, 32, u32, block, cta,

    relaxed, exch, 64, u64, block, cta,
    acquire, exch, 64, u64, block, cta,
    release, exch, 64, u64, block, cta,
    acqrel,  exch, 64, u64, block, cta,
    volatile, exch, 64, u64, block, cta,

    relaxed, exch, 32, u32, system, sys,
    acquire, exch, 32, u32, system, sys,
    release, exch, 32, u32, system, sys,
    acqrel,  exch, 32, u32, system, sys,
    volatile, exch, 32, u32, system, sys,

    relaxed, exch, 64, u64, system, sys,
    acquire, exch, 64, u64, system, sys,
    release, exch, 64, u64, system, sys,
    acqrel,  exch, 64, u64, system, sys,
    volatile, exch, 64, u64, system, sys,

    relaxed, exch, 32, i32, device, gpu,
    acquire, exch, 32, i32, device, gpu,
    release, exch, 32, i32, device, gpu,
    acqrel,  exch, 32, i32, device, gpu,
    volatile, exch, 32, i32, device, gpu,

    relaxed, exch, 64, i64, device, gpu,
    acquire, exch, 64, i64, device, gpu,
    release, exch, 64, i64, device, gpu,
    acqrel,  exch, 64, i64, device, gpu,
    volatile, exch, 64, i64, device, gpu,

    relaxed, exch, 32, i32, block, cta,
    acquire, exch, 32, i32, block, cta,
    release, exch, 32, i32, block, cta,
    acqrel,  exch, 32, i32, block, cta,
    volatile, exch, 32, i32, block, cta,

    relaxed, exch, 64, i64, block, cta,
    acquire, exch, 64, i64, block, cta,
    release, exch, 64, i64, block, cta,
    acqrel,  exch, 64, i64, block, cta,
    volatile, exch, 64, i64, block, cta,

    relaxed, exch, 32, i32, system, sys,
    acquire, exch, 32, i32, system, sys,
    release, exch, 32, i32, system, sys,
    acqrel,  exch, 32, i32, system, sys,
    volatile, exch, 32, i32, system, sys,

    relaxed, exch, 64, i64, system, sys,
    acquire, exch, 64, i64, system, sys,
    release, exch, 64, i64, system, sys,
    acqrel,  exch, 64, i64, system, sys,
    volatile, exch, 64, i64, system, sys,

    relaxed, exch, 32, f32, device, gpu,
    acquire, exch, 32, f32, device, gpu,
    release, exch, 32, f32, device, gpu,
    acqrel,  exch, 32, f32, device, gpu,
    volatile, exch, 32, f32, device, gpu,

    relaxed, exch, 64, f64, device, gpu,
    acquire, exch, 64, f64, device, gpu,
    release, exch, 64, f64, device, gpu,
    acqrel,  exch, 64, f64, device, gpu,
    volatile, exch, 64, f64, device, gpu,

    relaxed, exch, 32, f32, block, cta,
    acquire, exch, 32, f32, block, cta,
    release, exch, 32, f32, block, cta,
    acqrel,  exch, 32, f32, block, cta,
    volatile, exch, 32, f32, block, cta,

    relaxed, exch, 64, f64, block, cta,
    acquire, exch, 64, f64, block, cta,
    release, exch, 64, f64, block, cta,
    acqrel,  exch, 64, f64, block, cta,
    volatile, exch, 64, f64, block, cta,

    relaxed, exch, 32, f32, system, sys,
    acquire, exch, 32, f32, system, sys,
    release, exch, 32, f32, system, sys,
    acqrel,  exch, 32, f32, system, sys,
    volatile, exch, 32, f32, system, sys,

    relaxed, exch, 64, f64, system, sys,
    acquire, exch, 64, f64, system, sys,
    release, exch, 64, f64, system, sys,
    acqrel,  exch, 64, f64, system, sys,
    volatile, exch, 64, f64, system, sys,
}

macro_rules! atomic_fetch_op_3_reg {
    ($($ordering:ident, $op:ident, $width:literal, $type:ty, $scope:ident, $scope_asm:ident),* $(,)*) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                #[doc = concat!(
                    "Fetches the value in ptr, performs a ",
                    stringify!($op),
                    ", and returns the original value"
                )]
                pub unsafe fn [<atomic_fetch_ $op _ $ordering _ $type _ $scope>](ptr: *mut $type, val: $type) -> $type {
                    let mut out;
                    asm!(
                        concat!(
                            "atom.",
                            ordering!($ordering),
                            stringify!($scope_asm),
                            ".",
                            stringify!($op),
                            ".",
                            ptx_type!($type),
                            " {}, [{}], {};"
                        ),
                        out([<reg $width>]) out,
                        in(reg64) ptr,
                        in([<reg $width>]) val,
                    );
                    out
                }
            }
        )*
    };
}

atomic_fetch_op_3_reg! {
    // add (unsigned)

    relaxed, add, 32, u32, device, gpu,
    acquire, add, 32, u32, device, gpu,
    release, add, 32, u32, device, gpu,
    acqrel,  add, 32, u32, device, gpu,
    volatile, add, 32, u32, device, gpu,

    relaxed, add, 64, u64, device, gpu,
    acquire, add, 64, u64, device, gpu,
    release, add, 64, u64, device, gpu,
    acqrel,  add, 64, u64, device, gpu,
    volatile, add, 64, u64, device, gpu,

    relaxed, add, 32, u32, block, cta,
    acquire, add, 32, u32, block, cta,
    release, add, 32, u32, block, cta,
    acqrel,  add, 32, u32, block, cta,
    volatile, add, 32, u32, block, cta,

    relaxed, add, 64, u64, block, cta,
    acquire, add, 64, u64, block, cta,
    release, add, 64, u64, block, cta,
    acqrel,  add, 64, u64, block, cta,
    volatile, add, 64, u64, block, cta,

    relaxed, add, 32, u32, system, sys,
    acquire, add, 32, u32, system, sys,
    release, add, 32, u32, system, sys,
    acqrel,  add, 32, u32, system, sys,
    volatile, add, 32, u32, system, sys,

    relaxed, add, 64, u64, system, sys,
    acquire, add, 64, u64, system, sys,
    release, add, 64, u64, system, sys,
    acqrel,  add, 64, u64, system, sys,
    volatile, add, 64, u64, system, sys,

    // add (signed)

    relaxed, add, 32, i32, device, gpu,
    acquire, add, 32, i32, device, gpu,
    release, add, 32, i32, device, gpu,
    acqrel,  add, 32, i32, device, gpu,
    volatile, add, 32, i32, device, gpu,

    relaxed, add, 64, i64, device, gpu,
    acquire, add, 64, i64, device, gpu,
    release, add, 64, i64, device, gpu,
    acqrel,  add, 64, i64, device, gpu,
    volatile, add, 64, i64, device, gpu,

    relaxed, add, 32, i32, block, cta,
    acquire, add, 32, i32, block, cta,
    release, add, 32, i32, block, cta,
    acqrel,  add, 32, i32, block, cta,
    volatile, add, 32, i32, block, cta,

    relaxed, add, 64, i64, block, cta,
    acquire, add, 64, i64, block, cta,
    release, add, 64, i64, block, cta,
    acqrel,  add, 64, i64, block, cta,
    volatile, add, 64, i64, block, cta,

    relaxed, add, 32, i32, system, sys,
    acquire, add, 32, i32, system, sys,
    release, add, 32, i32, system, sys,
    acqrel,  add, 32, i32, system, sys,
    volatile, add, 32, i32, system, sys,

    relaxed, add, 64, i64, system, sys,
    acquire, add, 64, i64, system, sys,
    release, add, 64, i64, system, sys,
    acqrel,  add, 64, i64, system, sys,
    volatile, add, 64, i64, system, sys,

    // add (float)

    relaxed, add, 32, f32, device, gpu,
    acquire, add, 32, f32, device, gpu,
    release, add, 32, f32, device, gpu,
    acqrel,  add, 32, f32, device, gpu,
    volatile, add, 32, f32, device, gpu,

    relaxed, add, 64, f64, device, gpu,
    acquire, add, 64, f64, device, gpu,
    release, add, 64, f64, device, gpu,
    acqrel,  add, 64, f64, device, gpu,
    volatile, add, 64, f64, device, gpu,

    relaxed, add, 32, f32, block, cta,
    acquire, add, 32, f32, block, cta,
    release, add, 32, f32, block, cta,
    acqrel,  add, 32, f32, block, cta,
    volatile, add, 32, f32, block, cta,

    relaxed, add, 64, f64, block, cta,
    acquire, add, 64, f64, block, cta,
    release, add, 64, f64, block, cta,
    acqrel,  add, 64, f64, block, cta,
    volatile, add, 64, f64, block, cta,

    relaxed, add, 32, f32, system, sys,
    acquire, add, 32, f32, system, sys,
    release, add, 32, f32, system, sys,
    acqrel,  add, 32, f32, system, sys,
    volatile, add, 32, f32, system, sys,

    relaxed, add, 64, f64, system, sys,
    acquire, add, 64, f64, system, sys,
    release, add, 64, f64, system, sys,
    acqrel,  add, 64, f64, system, sys,
    volatile, add, 64, f64, system, sys,

    // min (unsigned)

    relaxed, min, 32, u32, device, gpu,
    acquire, min, 32, u32, device, gpu,
    release, min, 32, u32, device, gpu,
    acqrel,  min, 32, u32, device, gpu,
    volatile, min, 32, u32, device, gpu,

    relaxed, min, 64, u64, device, gpu,
    acquire, min, 64, u64, device, gpu,
    release, min, 64, u64, device, gpu,
    acqrel,  min, 64, u64, device, gpu,
    volatile, min, 64, u64, device, gpu,

    relaxed, min, 32, u32, block, cta,
    acquire, min, 32, u32, block, cta,
    release, min, 32, u32, block, cta,
    acqrel,  min, 32, u32, block, cta,
    volatile, min, 32, u32, block, cta,

    relaxed, min, 64, u64, block, cta,
    acquire, min, 64, u64, block, cta,
    release, min, 64, u64, block, cta,
    acqrel,  min, 64, u64, block, cta,
    volatile, min, 64, u64, block, cta,

    relaxed, min, 32, u32, system, sys,
    acquire, min, 32, u32, system, sys,
    release, min, 32, u32, system, sys,
    acqrel,  min, 32, u32, system, sys,
    volatile, min, 32, u32, system, sys,

    relaxed, min, 64, u64, system, sys,
    acquire, min, 64, u64, system, sys,
    release, min, 64, u64, system, sys,
    acqrel,  min, 64, u64, system, sys,
    volatile, min, 64, u64, system, sys,

    // min (signed)

    relaxed, min, 32, i32, device, gpu,
    acquire, min, 32, i32, device, gpu,
    release, min, 32, i32, device, gpu,
    acqrel,  min, 32, i32, device, gpu,
    volatile, min, 32, i32, device, gpu,

    relaxed, min, 64, i64, device, gpu,
    acquire, min, 64, i64, device, gpu,
    release, min, 64, i64, device, gpu,
    acqrel,  min, 64, i64, device, gpu,
    volatile, min, 64, i64, device, gpu,

    relaxed, min, 32, i32, block, cta,
    acquire, min, 32, i32, block, cta,
    release, min, 32, i32, block, cta,
    acqrel,  min, 32, i32, block, cta,
    volatile, min, 32, i32, block, cta,

    relaxed, min, 64, i64, block, cta,
    acquire, min, 64, i64, block, cta,
    release, min, 64, i64, block, cta,
    acqrel,  min, 64, i64, block, cta,
    volatile, min, 64, i64, block, cta,

    relaxed, min, 32, i32, system, sys,
    acquire, min, 32, i32, system, sys,
    release, min, 32, i32, system, sys,
    acqrel,  min, 32, i32, system, sys,
    volatile, min, 32, i32, system, sys,

    relaxed, min, 64, i64, system, sys,
    acquire, min, 64, i64, system, sys,
    release, min, 64, i64, system, sys,
    acqrel,  min, 64, i64, system, sys,
    volatile, min, 64, i64, system, sys,

    // max (unsigned)

    relaxed, max, 32, u32, device, gpu,
    acquire, max, 32, u32, device, gpu,
    release, max, 32, u32, device, gpu,
    acqrel,  max, 32, u32, device, gpu,
    volatile, max, 32, u32, device, gpu,

    relaxed, max, 64, u64, device, gpu,
    acquire, max, 64, u64, device, gpu,
    release, max, 64, u64, device, gpu,
    acqrel,  max, 64, u64, device, gpu,
    volatile, max, 64, u64, device, gpu,

    relaxed, max, 32, u32, block, cta,
    acquire, max, 32, u32, block, cta,
    release, max, 32, u32, block, cta,
    acqrel,  max, 32, u32, block, cta,
    volatile, max, 32, u32, block, cta,

    relaxed, max, 64, u64, block, cta,
    acquire, max, 64, u64, block, cta,
    release, max, 64, u64, block, cta,
    acqrel,  max, 64, u64, block, cta,
    volatile, max, 64, u64, block, cta,

    relaxed, max, 32, u32, system, sys,
    acquire, max, 32, u32, system, sys,
    release, max, 32, u32, system, sys,
    acqrel,  max, 32, u32, system, sys,
    volatile, max, 32, u32, system, sys,

    relaxed, max, 64, u64, system, sys,
    acquire, max, 64, u64, system, sys,
    release, max, 64, u64, system, sys,
    acqrel,  max, 64, u64, system, sys,
    volatile, max, 64, u64, system, sys,

    // max (signed)

    relaxed, max, 32, i32, device, gpu,
    acquire, max, 32, i32, device, gpu,
    release, max, 32, i32, device, gpu,
    acqrel,  max, 32, i32, device, gpu,
    volatile, max, 32, i32, device, gpu,

    relaxed, max, 64, i64, device, gpu,
    acquire, max, 64, i64, device, gpu,
    release, max, 64, i64, device, gpu,
    acqrel,  max, 64, i64, device, gpu,
    volatile, max, 64, i64, device, gpu,

    relaxed, max, 32, i32, block, cta,
    acquire, max, 32, i32, block, cta,
    release, max, 32, i32, block, cta,
    acqrel,  max, 32, i32, block, cta,
    volatile, max, 32, i32, block, cta,

    relaxed, max, 64, i64, block, cta,
    acquire, max, 64, i64, block, cta,
    release, max, 64, i64, block, cta,
    acqrel,  max, 64, i64, block, cta,
    volatile, max, 64, i64, block, cta,

    relaxed, max, 32, i32, system, sys,
    acquire, max, 32, i32, system, sys,
    release, max, 32, i32, system, sys,
    acqrel,  max, 32, i32, system, sys,
    volatile, max, 32, i32, system, sys,

    relaxed, max, 64, i64, system, sys,
    acquire, max, 64, i64, system, sys,
    release, max, 64, i64, system, sys,
    acqrel,  max, 64, i64, system, sys,
    volatile, max, 64, i64, system, sys,
}

macro_rules! atomic_fetch_op_4_reg {
    ($($ordering:ident, $op:ident, $width:literal, $type:ty, $scope:ident, $scope_asm:ident),* $(,)*) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                #[doc = concat!(
                    "Fetches the value in ptr, performs a ",
                    stringify!($op),
                    ", and returns the original value"
                )]
                pub unsafe fn [<atomic_fetch_ $op _ $ordering _ $type _ $scope>](ptr: *mut $type, first_val: $type, second_val: $type) -> $type {
                    let mut out;
                    asm!(
                        concat!(
                            "atom.",
                            ordering!($ordering),
                            stringify!($scope_asm),
                            ".",
                            stringify!($op),
                            ".",
                            ptx_type!($type),
                            " {}, [{}], {}, {};"
                        ),
                        out([<reg $width>]) out,
                        in(reg64) ptr,
                        in([<reg $width>]) first_val,
                        in([<reg $width>]) second_val,
                    );
                    out
                }
            }
        )*
    };
}

atomic_fetch_op_4_reg! {
    // compare and swap

    relaxed, cas, 32, u32, device, gpu,
    acquire, cas, 32, u32, device, gpu,
    release, cas, 32, u32, device, gpu,
    acqrel,  cas, 32, u32, device, gpu,
    volatile, cas, 32, u32, device, gpu,

    relaxed, cas, 64, u64, device, gpu,
    acquire, cas, 64, u64, device, gpu,
    release, cas, 64, u64, device, gpu,
    acqrel,  cas, 64, u64, device, gpu,
    volatile, cas, 64, u64, device, gpu,

    relaxed, cas, 32, u32, block, cta,
    acquire, cas, 32, u32, block, cta,
    release, cas, 32, u32, block, cta,
    acqrel,  cas, 32, u32, block, cta,
    volatile, cas, 32, u32, block, cta,

    relaxed, cas, 64, u64, block, cta,
    acquire, cas, 64, u64, block, cta,
    release, cas, 64, u64, block, cta,
    acqrel,  cas, 64, u64, block, cta,
    volatile, cas, 64, u64, block, cta,

    relaxed, cas, 32, u32, system, sys,
    acquire, cas, 32, u32, system, sys,
    release, cas, 32, u32, system, sys,
    acqrel,  cas, 32, u32, system, sys,
    volatile, cas, 32, u32, system, sys,

    relaxed, cas, 64, u64, system, sys,
    acquire, cas, 64, u64, system, sys,
    release, cas, 64, u64, system, sys,
    acqrel,  cas, 64, u64, system, sys,
    volatile, cas, 64, u64, system, sys,

    relaxed, cas, 32, i32, device, gpu,
    acquire, cas, 32, i32, device, gpu,
    release, cas, 32, i32, device, gpu,
    acqrel,  cas, 32, i32, device, gpu,
    volatile, cas, 32, i32, device, gpu,

    relaxed, cas, 64, i64, device, gpu,
    acquire, cas, 64, i64, device, gpu,
    release, cas, 64, i64, device, gpu,
    acqrel,  cas, 64, i64, device, gpu,
    volatile, cas, 64, i64, device, gpu,

    relaxed, cas, 32, i32, block, cta,
    acquire, cas, 32, i32, block, cta,
    release, cas, 32, i32, block, cta,
    acqrel,  cas, 32, i32, block, cta,
    volatile, cas, 32, i32, block, cta,

    relaxed, cas, 64, i64, block, cta,
    acquire, cas, 64, i64, block, cta,
    release, cas, 64, i64, block, cta,
    acqrel,  cas, 64, i64, block, cta,
    volatile, cas, 64, i64, block, cta,

    relaxed, cas, 32, i32, system, sys,
    acquire, cas, 32, i32, system, sys,
    release, cas, 32, i32, system, sys,
    acqrel,  cas, 32, i32, system, sys,
    volatile, cas, 32, i32, system, sys,

    relaxed, cas, 64, i64, system, sys,
    acquire, cas, 64, i64, system, sys,
    release, cas, 64, i64, system, sys,
    acqrel,  cas, 64, i64, system, sys,
    volatile, cas, 64, i64, system, sys,

    relaxed, cas, 32, f32, device, gpu,
    acquire, cas, 32, f32, device, gpu,
    release, cas, 32, f32, device, gpu,
    acqrel,  cas, 32, f32, device, gpu,
    volatile, cas, 32, f32, device, gpu,

    relaxed, cas, 64, f64, device, gpu,
    acquire, cas, 64, f64, device, gpu,
    release, cas, 64, f64, device, gpu,
    acqrel,  cas, 64, f64, device, gpu,
    volatile, cas, 64, f64, device, gpu,

    relaxed, cas, 32, f32, block, cta,
    acquire, cas, 32, f32, block, cta,
    release, cas, 32, f32, block, cta,
    acqrel,  cas, 32, f32, block, cta,
    volatile, cas, 32, f32, block, cta,

    relaxed, cas, 64, f64, block, cta,
    acquire, cas, 64, f64, block, cta,
    release, cas, 64, f64, block, cta,
    acqrel,  cas, 64, f64, block, cta,
    volatile, cas, 64, f64, block, cta,

    relaxed, cas, 32, f32, system, sys,
    acquire, cas, 32, f32, system, sys,
    release, cas, 32, f32, system, sys,
    acqrel,  cas, 32, f32, system, sys,
    volatile, cas, 32, f32, system, sys,

    relaxed, cas, 64, f64, system, sys,
    acquire, cas, 64, f64, system, sys,
    release, cas, 64, f64, system, sys,
    acqrel,  cas, 64, f64, system, sys,
    volatile, cas, 64, f64, system, sys,
}

#[allow(unused_macros)]
macro_rules! negation {
    (u32, $val:ident) => {{ -($val as i32) }};
    (u64, $val:ident) => {{ -($val as i64) }};
    ($type:ty, $val:ident) => {{ -$val }};
}

// atomic sub is a little special, nvcc implements it as an atomic add with a negated operand. PTX
// does not have atom.sub.
macro_rules! atomic_sub {
    ($($ordering:ident, $width:literal, $type:ty, $scope:ident, $scope_asm:ident),* $(,)*) => {
        $(
            paste! {
                #[$crate::gpu_only]
                #[allow(clippy::missing_safety_doc)]
                /// Fetches the value in ptr, performs a sub, and returns the original value
                pub unsafe fn [<atomic_fetch_sub_ $ordering _ $type _ $scope>](ptr: *mut $type, val: $type) -> $type {
                    let mut out;
                    asm!(
                        concat!(
                            "atom.",
                            ordering!($ordering),
                            stringify!($scope_asm),
                            ".",
                            "add",
                            ".",
                            ptx_type!($type),
                            " {}, [{}], {};"
                        ),
                        out([<reg $width>]) out,
                        in(reg64) ptr,
                        in([<reg $width>]) negation!($type, val),
                    );
                    out
                }
            }
        )*
    };
}

atomic_sub! {
    // sub (unsigned)

    relaxed, 32, u32, device, gpu,
    acquire, 32, u32, device, gpu,
    release, 32, u32, device, gpu,
    acqrel,  32, u32, device, gpu,
    volatile,  32, u32, device, gpu,

    relaxed, 64, u64, device, gpu,
    acquire, 64, u64, device, gpu,
    release, 64, u64, device, gpu,
    acqrel,  64, u64, device, gpu,
    volatile,  64, u64, device, gpu,

    relaxed, 32, u32, block, cta,
    acquire, 32, u32, block, cta,
    release, 32, u32, block, cta,
    acqrel,  32, u32, block, cta,
    volatile,  32, u32, block, cta,

    relaxed, 64, u64, block, cta,
    acquire, 64, u64, block, cta,
    release, 64, u64, block, cta,
    acqrel,  64, u64, block, cta,
    volatile,  64, u64, block, cta,

    relaxed, 32, u32, system, sys,
    acquire, 32, u32, system, sys,
    release, 32, u32, system, sys,
    acqrel,  32, u32, system, sys,
    volatile,  32, u32, system, sys,

    relaxed, 64, u64, system, sys,
    acquire, 64, u64, system, sys,
    release, 64, u64, system, sys,
    acqrel,  64, u64, system, sys,
    volatile,  64, u64, system, sys,

    // sub (signed)

    relaxed, 32, i32, device, gpu,
    acquire, 32, i32, device, gpu,
    release, 32, i32, device, gpu,
    acqrel,  32, i32, device, gpu,
    volatile,  32, i32, device, gpu,

    relaxed, 64, i64, device, gpu,
    acquire, 64, i64, device, gpu,
    release, 64, i64, device, gpu,
    acqrel,  64, i64, device, gpu,
    volatile,  64, i64, device, gpu,

    relaxed, 32, i32, block, cta,
    acquire, 32, i32, block, cta,
    release, 32, i32, block, cta,
    acqrel,  32, i32, block, cta,
    volatile,  32, i32, block, cta,

    relaxed, 64, i64, block, cta,
    acquire, 64, i64, block, cta,
    release, 64, i64, block, cta,
    acqrel,  64, i64, block, cta,
    volatile,  64, i64, block, cta,

    relaxed, 32, i32, system, sys,
    acquire, 32, i32, system, sys,
    release, 32, i32, system, sys,
    acqrel,  32, i32, system, sys,
    volatile,  32, i32, system, sys,

    relaxed, 64, i64, system, sys,
    acquire, 64, i64, system, sys,
    release, 64, i64, system, sys,
    acqrel,  64, i64, system, sys,
    volatile,  64, i64, system, sys,

    // sub (float)

    relaxed, 32, f32, device, gpu,
    acquire, 32, f32, device, gpu,
    release, 32, f32, device, gpu,
    acqrel,  32, f32, device, gpu,
    volatile,  32, f32, device, gpu,

    relaxed, 64, f64, device, gpu,
    acquire, 64, f64, device, gpu,
    release, 64, f64, device, gpu,
    acqrel,  64, f64, device, gpu,
    volatile,  64, f64, device, gpu,

    relaxed, 32, f32, block, cta,
    acquire, 32, f32, block, cta,
    release, 32, f32, block, cta,
    acqrel,  32, f32, block, cta,
    volatile,  32, f32, block, cta,

    relaxed, 64, f64, block, cta,
    acquire, 64, f64, block, cta,
    release, 64, f64, block, cta,
    acqrel,  64, f64, block, cta,
    volatile,  64, f64, block, cta,

    relaxed, 32, f32, system, sys,
    acquire, 32, f32, system, sys,
    release, 32, f32, system, sys,
    acqrel,  32, f32, system, sys,
    volatile,  32, f32, system, sys,

    relaxed, 64, f64, system, sys,
    acquire, 64, f64, system, sys,
    release, 64, f64, system, sys,
    acqrel,  64, f64, system, sys,
    volatile,  64, f64, system, sys,
}
