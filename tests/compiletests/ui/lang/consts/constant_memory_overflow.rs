// Test that automatic constant memory placement spills to global with a warning
// when exceeding the 64KB limit

// compile-flags: -Cllvm-args=--use-constant-memory-space

#![no_std]
use cuda_std::*;

// 35KB per array, 3 arrays = 105KB total (well above 64KB limit)
const ARRAY_SIZE: usize = 35 * 1024 / 4;

// NO explicit address_space - let the automatic placement handle it
// With use_constant_memory_space=true, these should try to go to constant memory
static BIG_ARRAY_1: [u32; ARRAY_SIZE] = [111u32; ARRAY_SIZE];
static BIG_ARRAY_2: [u32; ARRAY_SIZE] = [222u32; ARRAY_SIZE];
static BIG_ARRAY_3: [u32; ARRAY_SIZE] = [333u32; ARRAY_SIZE];

#[kernel]
pub unsafe fn test_kernel(out: *mut u32) {
    unsafe { *out = BIG_ARRAY_1[0] + BIG_ARRAY_2[0] + BIG_ARRAY_3[0] };
}
