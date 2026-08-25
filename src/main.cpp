#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/kernel_00_naive.cuh"
#include "../include/kernel_01_shared.cuh"
#include "../include/kernel_02_register.cuh"
#include "../include/kernel_03_unrolled.cuh"
#include "../include/kernel_04_tensor.cuh"

void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    
}

bool verify_result(const float* host_C, const float* device_C, int size) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(host_C[i] - device_C[i]) > 0.01) {
            std::cout << "Mismatch at index " << i << ": CPU=" << host_C[i] << " GPU=" << device_C[i] << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    size_t bytes_FP32 = M * N * sizeof(float);
    size_t bytes_FP16 = M * N * sizeof(half);

    std::cout << "Running CUDA GEMM Suite (Matrix Size: " << M << "x" << N << "x" << K << ")\n\n";

    std::vector<float> h_A(M * K, 1.0f);
    std::vector<float> h_B(K * N, 2.0f);
    std::vector<float> h_C_cpu(M * N, 0.0f);
    std::vector<float> h_C_gpu(M * N, 0.0f);

    std::vector<half> h_A_fp16(M * K);
    std::vector<half> h_B_fp16(K * N);
    for (int i = 0; i < M * K; ++i) h_A_fp16[i] = __float2half(h_A[i]);
    for (int i = 0; i < K * N; ++i) h_B_fp16[i] = __float2half(h_B[i]);

    float *d_A, *d_B, *d_C;
    half *d_A_fp16, *d_B_fp16;

    cudaMalloc(&d_A, bytes_FP32);
    cudaMalloc(&d_B, bytes_FP32);
    cudaMalloc(&d_C, bytes_FP32);

    cudaMalloc(&d_A_fp16, bytes_FP16);
    cudaMalloc(&d_B_fp16, bytes_FP16);

    cudaMemcpy(d_A, h_A.data(), bytes_FP32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes_FP32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_fp16, h_A_fp16.data(), bytes_FP16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, h_B_fp16.data(), bytes_FP16, cudaMemcpyHostToDevice);

    cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    #define RUN_BENCHMARK(name, ...) \
    { \
        __VA_ARGS__; \
        cudaDeviceSynchronize(); \
        \
        cudaEventRecord(start); \
        for (int iter = 0; iter < 10; ++iter) { \
            __VA_ARGS__; \
        } \
        cudaEventRecord(stop); \
        cudaEventSynchronize(stop); \
        \
        float milliseconds = 0; \
        cudaEventElapsedTime(&milliseconds, start, stop); \
        float avg_ms = milliseconds / 10.0f; \
        \
        double flops = 2.0 * M * N * K; \
        double tflops = (flops / (avg_ms / 1000.0)) / 1e12; \
        \
        cudaMemcpy(h_C_gpu.data(), d_C, bytes_FP32, cudaMemcpyDeviceToHost); \
        bool correct = verify_result(h_C_cpu.data(), h_C_gpu.data(), M * N); \
        \
        std::cout << "[" << name << "]\n"; \
        std::cout << "  Avg Time: " << avg_ms << " ms\n"; \
        std::cout << "  Performance: " << tflops << " TFLOPS\n"; \
        std::cout << "  Status: " << (correct ? "PASSED" : "FAILED") << "\n\n"; \
    }

    // Naive
    {
        dim3 block0(16, 16);
        dim3 grid0((N + block0.x - 1) / block0.x, (M + block0.y - 1) / block0.y);
        RUN_BENCHMARK("Kernel 0: Naive GEMM", gemm_naive<<<grid0, block0>>>(d_A, d_B, d_C, M, N, K));
    }

    // Shared Memory Tiling
    {
        dim3 block1(32, 32);
        dim3 grid1((N + block1.x - 1) / block1.x, (M + block1.y - 1) / block1.y);
        RUN_BENCHMARK("Kernel 1: Shared Memory Tiling", gemm_shared_memory<<<grid1, block1>>>(d_A, d_B, d_C, M, N, K));
    }

    // Register Tiling
    {
        dim3 block2(BN / TN, BM / TM);
        dim3 grid2((N + BN - 1) / BN, (M + BM - 1) / BM);
        RUN_BENCHMARK("Kernel 2: Register Tiling", gemm_register_tiling<<<grid2, block2>>>(d_A, d_B, d_C, M, N, K));
    }

    // Unrolled Loop
    {
        dim3 block1(32, 32);
        dim3 grid1((N + block1.x - 1) / block1.x, (M + block1.y - 1) / block1.y);
        RUN_BENCHMARK("Kernel 3: Unrolled Inner Loop", gemm_unrolled<<<grid1, block1>>>(d_A, d_B, d_C, M, N, K));
    }

    // WMMA Tensor Cores
    {
        dim3 block4(32, 4);
        dim3 grid4((N + (WMMA_N * block4.x / 32) - 1) / (WMMA_N * block4.x / 32),
                   (M + (WMMA_M * block4.y) - 1) / (WMMA_M * block4.y));
        RUN_BENCHMARK("Kernel 4: WMMA FP16 Tensor Cores", gemm_wmma_fp16<<<grid4, block4>>>(d_A_fp16, d_B_fp16, d_C, M, N, K));
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
