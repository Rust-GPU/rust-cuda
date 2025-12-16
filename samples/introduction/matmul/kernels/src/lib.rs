use core::mem::MaybeUninit;
use cuda_std::*;

// SAFETY: This function is unsafe because it dereferences raw pointers.
#[kernel]
pub unsafe fn matrix_mul_cuda(c: *mut f32, a: &[f32], b: &[f32], wa: usize, wb: usize) {
    let bx: usize = cuda_std::thread::block_idx().x as usize;
    let by: usize = cuda_std::thread::block_idx().y as usize;

    let tx: usize = cuda_std::thread::thread_idx().x as usize;
    let ty: usize = cuda_std::thread::thread_idx().y as usize;

    const BLOCK_SIZE: usize = 32;
    let a_begin = wa * BLOCK_SIZE * by;
    let a_end = a_begin + wa - 1;
    let a_step = BLOCK_SIZE;

    let b_begin = BLOCK_SIZE * bx;
    let b_step = BLOCK_SIZE * wb;

    let mut c_sub: f32 = 0.0;
    let mut kahan_correction_factor = 0.0f32;
    let mut bi = b_begin;

    for ai in (a_begin..=a_end).step_by(a_step) {
        // The equivalent Cuda C++ code for the below is:
        // ```
        // __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        // ```
        // This memory space is shared between threads of the same block
        #[address_space(shared)]
        static mut As: [[MaybeUninit<f32>; BLOCK_SIZE]; BLOCK_SIZE] =
            [[const { MaybeUninit::uninit() }; BLOCK_SIZE]; BLOCK_SIZE];

        #[address_space(shared)]
        static mut Bs: [[MaybeUninit<f32>; BLOCK_SIZE]; BLOCK_SIZE] =
            [[const { MaybeUninit::uninit() }; BLOCK_SIZE]; BLOCK_SIZE];

        // Load A and B matrices into shared memory
        // A.add(index) returns the pointer to the index-th element of A
        // Hence a dereference is needed to get the value at that index
        unsafe {
            As[ty][tx].write(a[ai + wa * ty + tx]);
            Bs[ty][tx].write(b[bi + wb * ty + tx]);
        }

        // Synchronize to make sure the matrices are loaded
        cuda_std::thread::sync_threads();
        for k in 0..BLOCK_SIZE {
            // Typically, this would be a simple calculation:
            // ```
            // c_sub += As[ty][k] * Bs[k][tx];
            // ```
            // However, to improve numerical stability, we use Kahan summation here so that the error can be isolated
            // and not allow it to accumulate in c_sub
            let input = unsafe { As[ty][k].assume_init() * Bs[k][tx].assume_init() };
            let y = input - kahan_correction_factor;
            let sum = c_sub + y;

            // This seems like the correction factor would yield zero, however due to f32 precision limitations,
            // it helps to isolate the small errors that would otherwise accumulate in c_sub
            kahan_correction_factor = (sum - c_sub) - y;
            c_sub = sum;
        }

        // Synchronize to make sure that the preceding computation is done before loading two new sub-matrices of A and B in the next iteration
        cuda_std::thread::sync_threads();

        bi += b_step;
    }

    let ci = wb * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    unsafe {
        *c.add((ci + wb * ty + tx) as usize) = c_sub;
    }
}
