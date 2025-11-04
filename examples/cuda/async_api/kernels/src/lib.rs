use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn increment(g_data: *mut u32, inc_value: u32) {
    // This can also be obtained directly as
    //
    // let idx: usize = cuda_std::thread::index() as usize;
    let idx: usize = (cuda_std::thread::block_dim().x * cuda_std::thread::block_idx().x
        + cuda_std::thread::thread_idx().x) as usize;

    let elem: &mut u32 = unsafe { &mut *g_data.add(idx) };
    *elem = *elem + inc_value;
}
