# Custom Cuda General Matrix Multiplication and Kernel Optimization
Comparing different implementations of general matrix multiplication (GEMM). This repository explores low-level GPU microarchitecture bottlenecks (DRAM bandwidth, shared memory bank conflicts, instruction-level parallelism) and validates optimizations empirically using NVIDIA's Nsight Compute CLI (`ncu`).

---

## Performance Summary

*Evaluated on an NVIDIA Tesla T4 (Compute Capability 7.5) on Google Colab, with* $1024 \times 1024 \times 1024$ *FP32/FP16 matrix multiplication.*

| Kernel Stage | Optimization Applied | Latency (ms) | Throughput (TFLOPS) | Speedup vs Naive |
| :--- | :--- | :--- | :--- | :--- |
| **Kernel 0** | Naive Global Memory Access | 9.19 ms | 0.23 TFLOPS | 1.0x (Baseline) |
| **Kernel 1** | Shared Memory Tiling ($32 \times 32$) | 6.23 ms | 0.34 TFLOPS | 1.47x |
| **Kernel 2** | Register Tiling ($2 \times 2$ Thread Tile) | 2.18 ms | 0.98 TFLOPS | 4.21x |
| **Kernel 3** | Loop Unrolling (`#pragma unroll`) | 3.65 ms | 0.58 TFLOPS | 2.51x |
| **Kernel 4** | FP16 WMMA (Tensor Cores) | 0.074 ms | 29.02 TFLOPS | 124x |

---

## Low-Level Profiling Insights (Nsight Compute)

Using `ncu`, hardware performance counters were logged to verify memory hierarchy behavior.

For Global Memory Bandwidth Optimization, the naive kernel had ~134.2 Million L1/TEX global load sectors requested , while shared memory tiling reduced global load sectors to ~8.38 Million, a ~16x reduction in DRAM transaction overhead.

For Compute Throughput & Occupancy, Kernel 1 Tiling achieved ~89.38% Active SM Throughput and ~87.87% Memory Pipeline Utilization.

---

## Directory Structure

```text
.
├── CMakeLists.txt
├── include/
│   ├── kernel_00_naive.cuh
│   ├── kernel_01_shared.cuh
│   ├── kernel_02_register.cuh
│   ├── kernel_03_unrolled.cuh
│   └── kernel_04_tensor.cuh
├── src/
│   └── main.cpp
└── README.md
```
