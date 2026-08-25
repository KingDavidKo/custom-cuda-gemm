#ifndef KERNEL_02_REGISTER_CUH
#define KERNEL_02_REGISTER_CUH

#include <cuda_runtime.h>

// Standard block configuration for register tiling
// Thread block size: (BN / TN) x (BM / TM) = (32/4) x (32/4) = 8 x 8 = 64 threads per block
#define BM 32
#define BN 32
#define BK 16
#define TM 4
#define TN 4

__global__ void gemm_register_tiling(const float* __restrict__ A, 
                                     const float* __restrict__ B, 
                                     float* __restrict__ C, 
                                     int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    float rC[TM][TN] = {0.0f};

    const int THREADS_X = BN / TN; // 8
    const int THREADS_Y = BM / TM; // 8
    const int total_threads = THREADS_X * THREADS_Y; // 64
    int tid = ty * THREADS_X + tx;

    for (int bk = 0; bk < K; bk += BK) {
        for (int i = tid; i < BM*BK; i += total_threads) {
            int row = i / BK;
            int col = i % BK;
            int global_row = by * BM + row;
            int global_col = bk + col;
            if (global_row < M && global_col < K) {
                sA[row][col] = A[global_row * K + global_col];
            } else {
                sA[row][col] = 0.0f;
            }
        }

        for (int i = tid; i < BK*BN; i += total_threads) {
            int row = i / BN;
            int col = i % BN;
            int global_row = bk + row;
            int global_col = bx * BN + col;
            if (global_row < K && global_col < N) {
                sB[row][col] = B[global_row*N + global_col];
            } else {
                sB[row][col] = 0.0f;
            }
        }

        __syncthreads();

        // MatMult register tiling
        for (int k = 0; k < BK; ++k) {
            float regA[TM];
            float regB[TN];

            for (int i = 0; i < TM; ++i) {
                regA[i] = sA[ty *TM + i][k];
            }
            for (int j = 0; j < TN; ++j) {
                regB[j] = sB[k][tx*TN + j];
            }


            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    rC[i][j] += regA[i] * regB[j];
                }
            }
        }


        __syncthreads();
    }

    // Global memory write-back
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int global_row = by*BM + ty*TM + i;
            int global_col = bx*BN + tx*TN + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = rC[i][j];
            }
        }
    }
}

#endif
