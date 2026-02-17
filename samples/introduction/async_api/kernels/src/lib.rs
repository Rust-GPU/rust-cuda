use cuda_std::prelude::*;

#[kernel]
/// # Safety
///
/// The user must ensure that the number of (threads * blocks * grids)
/// must not be greater than the number of elements in `g_data`.
pub unsafe fn increment(g_data: *mut u32, inc_value: u32) {
    // This can also be obtained directly as
    //
    // let idx: usize = cuda_std::thread::index();
    let idx: usize = cuda_std::thread::block_dim().x * cuda_std::thread::block_idx().x
        + cuda_std::thread::thread_idx().x;

    let elem: &mut u32 = unsafe { &mut *g_data.add(idx) };
    *elem += inc_value;
}
