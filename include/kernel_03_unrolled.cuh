
#ifndef KERNEL_03_UNROLLED_CUH
#define KERNEL_03_UNROLLED_CUH

#include <cuda_runtime.h>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

__global__ void gemm_unrolled(const float* __restrict__ A, 
                              const float* __restrict__ B, 
                              float* __restrict__ C, 
                              int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    for (int ph = 0; ph < (K + TILE_SIZE - 1) / TILE_SIZE; ++ph) {
        sA[threadIdx.y][threadIdx.x] = (row < M && (ph*TILE_SIZE + threadIdx.x) < K) ? A[row*K + ph*TILE_SIZE + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (col < N && (ph * TILE_SIZE + threadIdx.y) < K) ? B[(ph*TILE_SIZE + threadIdx.y) * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k<TILE_SIZE; ++k) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row*N + col] = acc;
    }
}

#endif
