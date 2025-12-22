// Here we use `unsigned long` to match the Rust version `usize`.
__global__ void sdot(const float *x, unsigned long x_n, const float *y, unsigned long y_n, float *out)
{

    extern __shared__ float shared_sum[];
    unsigned int i;

    unsigned int num_threads = gridDim.x * blockDim.x;
    unsigned int start_ind = blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;

    float sum = 0.0f;
    for (i = start_ind + tid; i < x_n; i += num_threads)
    {
        // Rust checks emulation
        if (i >= y_n)
            __trap();

        sum += x[i] * y[i];
    }
    shared_sum[tid] = sum;

    for (i = blockDim.x >> 1; i > 0; i >>= 1)
    {
        __syncthreads();
        if (tid < i)
        {
            shared_sum[tid] += shared_sum[tid + i];
        }
    }

    if (tid == 0)
    {
        out[blockIdx.x] = shared_sum[tid];
    }
}