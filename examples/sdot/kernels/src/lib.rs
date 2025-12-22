use cuda_std::{kernel, shared, thread};

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn sdot(x: &[f32], y: &[f32], out: *mut f32) {
    let shared_sum = shared::dynamic_shared_mem::<f32>();

    let num_threads = (thread::grid_dim_x() as usize) * (thread::block_dim_x() as usize);
    let start_ind = (thread::block_dim_x() as usize) * (thread::block_idx_x() as usize);
    let tid = thread::thread_idx_x() as usize;

    let mut sum = 0f32;
    for i in ((start_ind + tid)..x.len()).step_by(num_threads) {
        sum += x[i] * y[i];
    }
    unsafe {
        *shared_sum.add(tid) = sum;
    }

    let mut i = (thread::block_dim_x() >> 1) as usize;
    while i > 0 {
        thread::sync_threads();
        if tid < i {
            unsafe {
                *shared_sum.add(tid) += *shared_sum.add(tid + i);
            }
        }

        i >>= 1;
    }

    if tid == 0 {
        unsafe {
            *out.add(thread::block_idx_x() as usize) = *shared_sum.add(tid);
        }
    }
}
