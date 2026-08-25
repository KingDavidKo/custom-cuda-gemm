
#ifndef KERNEL_00_NAIVE_CUH
#define KERNEL_00_NAIVE_CUH

#include <cuda_runtime.h>

__global__ void gemm_naive(const float* __restrict__ A, 
                           const float* __restrict__ B, 
                           float* __restrict__ C, 
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; ++k) {
            val += A[row*K + k] * B[k*N + col];
        }
        C[row*N + col] = val;
    }
}

#endif
