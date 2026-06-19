#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t _e = (call);                                   \
    if (_e != cudaSuccess) {                                   \
        printf("CUDA error %s:%d: %s\n",                       \
               __FILE__, __LINE__, cudaGetErrorString(_e));    \
        return 1;                                              \
    }                                                          \
} while (0)

__global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 24;                 // ~16.7M elements
    const size_t sz = (size_t)n * sizeof(float);

    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, sz));
    CUDA_CHECK(cudaMallocManaged(&b, sz));
    CUDA_CHECK(cudaMallocManaged(&c, sz));
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    // warm-up launch (first launch includes one-time setup cost)
    vadd<<<blocks, threads>>>(a, b, c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed launch
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    vadd<<<blocks, threads>>>(a, b, c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // correctness
    bool ok = true;
    for (int i = 0; i < n; i++) { if (c[i] != 3.0f) { ok = false; break; } }

    // effective bandwidth: 2 reads + 1 write = 3 arrays moved
    double gb   = 3.0 * sz / 1e9;
    double gbps = gb / (ms / 1e3);

    printf("correct:             %s\n", ok ? "yes" : "NO");
    printf("time:                %.3f ms\n", ms);
    printf("effective bandwidth: %.1f GB/s\n", gbps);

    cudaFree(a); cudaFree(b); cudaFree(c);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
