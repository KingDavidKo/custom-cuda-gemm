
#ifndef KERNEL_01_SHARED_CUH
#define KERNEL_01_SHARED_CUH

#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void gemm_shared_memory(const float* __restrict__ A, 
                                   const float* __restrict__ B, 
                                   float* __restrict__ C, 
                                   int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ++ph) {
        if (row < M && (ph * TILE_SIZE + threadIdx.x) < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + ph * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (ph * TILE_SIZE + threadIdx.y) < K)
            sB[threadIdx.y][threadIdx.x] = B[(ph * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();


        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += sA[threadIdx.y][k]*sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row*N + col] = acc;
    }
}

#endif
