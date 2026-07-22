// build-pass
// compile-flags: -Cllvm-args=--disassemble-entry=shared_broadcast --error-format=human -Cdebuginfo=0

// Regression test: accesses to an `#[address_space(shared)]` static must be emitted in the
// shared address space, not as generic accesses behind an `addrspacecast` to addrspace 0.
//
// The kernel below must use `st.shared`/`ld.shared` on the raw symbol offset, and must contain
// no `cvta.shared` -- there is nothing here that needs a generic pointer.

use core::mem::MaybeUninit;
use cuda_std::{address_space, kernel, thread};

#[address_space(shared)]
static mut SH: [MaybeUninit<u32>; 32] = [MaybeUninit::uninit(); 32];

#[kernel]
pub unsafe fn shared_broadcast(out: *mut u32, n: u32) {
    let lane = thread::thread_idx_x() & 31;
    let warp = ((thread::thread_idx_x() >> 5) & 31) as usize;
    unsafe {
        if lane == 0 {
            SH[warp] = MaybeUninit::new(n);
        }
        thread::sync_threads();
        *out = SH[warp].assume_init();
    }
}
