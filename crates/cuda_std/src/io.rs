//! Utilities for printing to stdout from GPU threads.
//!
//! CUDA contains a syscall named `vprintf` which provides a way of atomically
//! printing from GPU threads, this module provides safe wrappers over it.
//! Printing is atomic, meaning that simultaneous calls from different
//! threads (which will naturally happen) will not clash with eachother
//! unlike printing from multiple CPU threads.
//!
//! # Important Notes
//!
//! Printing output in CUDA is stored inside of a circular buffer which has a fixed size (1mb by default).
//! If the buffer is filled, old output will be overwritten.
//!
//! This buffer is flushed for:
//! - Kernel launches
//! - Synchronization (stream/device synchronization)
//! - Blocking memory copies
//! - Module load/unload
//! - Context destruction
//!
//! This does NOT include exiting the program, however, because rust uses RAII, unless you leak the
//! context, output will always be flushed.

unsafe extern "C" {
    // CUDA syscalls implicitly defined by nvvm you can link to.

    #[doc(hidden)]
    pub fn vprintf(format: *const u8, valist: *const core::ffi::c_void) -> i32;

    #[doc(hidden)]
    pub fn __assertfail(
        message: *const u8,
        file: *const u8,
        line: u32,
        function: *const u8,
        char_size: usize,
    );
}

/// Alternative to [`print!`](std::print) which works on CUDA. See [`print`](self) for more info.
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {
        let msg = ::alloc::format!($($arg)*);
        let cstring = ::alloc::format!("{}\0", msg);
        unsafe {
            $crate::io::vprintf(cstring.as_ptr(), ::core::ptr::null_mut());
        }
    }
}

/// Alternative to [`println!`](std::println) which works on CUDA. See [`print`](self) for more info.
#[macro_export]
macro_rules! println {
    () => ($crate::print!("\n"));
    ($fmt:expr) => ($crate::print!(concat!($fmt, "\n")));
    ($fmt:expr, $($arg:tt)*) => ($crate::print!(concat!($fmt, "\n"), $($arg)*));
}

/// Asserts that two expression are equal and returns an `AssertionFailed` error to the application that launched the kernel
/// if it is not true.
#[macro_export]
macro_rules! assert_eq {
    ($a:expr, $b:expr) => {
        let _a = $a;
        let _b = $b;

        if _a != _b {
            let msg = ::alloc::format!(
                "\nassertion failed: ({} == {})\nleft : {:?}\nright: {:?}",
                stringify!($a),
                stringify!($b),
                _a,
                _b
            );

            unsafe {
                $crate::io::__assertfail(msg.as_ptr(), file!().as_ptr(), line!(), "".as_ptr(), 1)
            };
        }
    };
}

/// Asserts that two expression are not equal and returns an `AssertionFailed` error to the application that launched the kernel
/// if it is not true.
#[macro_export]
macro_rules! assert_ne {
    ($a:expr, $b:expr) => {
        let _a = $a;
        let _b = $b;

        if _a == _b {
            let msg = ::alloc::format!(
                "\nassertion failed: ({} != {})\nleft : {:?}\nright: {:?}",
                stringify!($a),
                stringify!($b),
                _a,
                _b
            );

            unsafe {
                $crate::io::__assertfail(msg.as_ptr(), file!().as_ptr(), line!(), "".as_ptr(), 1)
            };
        }
    };
}
