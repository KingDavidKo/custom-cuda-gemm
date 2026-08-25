#ifndef KERNEL_04_TENSOR_CUH
#define KERNEL_04_TENSOR_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;
using namespace nvcuda::wmma;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void gemm_wmma_fp16(const half* __restrict__ A, 
                               const half* __restrict__ B, 
                               float* __restrict__ C, 
                               int M, int N, int K) {
    int lda = K;
    int ldb = N;
    int ldc = N;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            load_matrix_sync(a_frag, A + aRow*lda + aCol, lda);
            load_matrix_sync(b_frag, B + bRow*ldb + bCol, ldb);

            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM*WMMA_M;
    int cCol = warpN*WMMA_N;


    if (cRow < M && cCol < N) {
        store_matrix_sync(C + cRow * ldc + cCol, c_frag, ldc, mem_row_major);
    }
}

#endif
