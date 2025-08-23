This is a technique report for this GEMM repository, which implements both normal and mixed precision GEMM kernels. The kernels are designed to optimize matrix multiplication operations using CUDA C++ and Triton, with a focus on performance enhancements through various optimization techniques.
 


# GEMM

## SM80 Normal BF16 GEMM Kernel

The SM80 kernel is designed to leverage the capabilities of the NVIDIA Ampere architecture for efficient matrix multiplication. The input matrix precision is bf16 while the output's is fp32.

- V1: tensor core instructions, no swizzling.
- V2: tensor core instructions, asynchronize smem operator.
- V3: tensor core instructions, asynchronize smem operator, swizzling.
- V4: tensor core instructions, asynchronize smem operator, swizzling, split-K
- V5: tensor core instructions, asynchronize smem operator, swizzling, split-K, pipeline-K


### V1 

The V1 is a simple version of GEMM. It uses the techniques below:
- Tensor core instructions
- Vectorized Memory Fetching

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=mma#wmma-description



The performance is:
- TFLOPS
| Problem Size | GEMM_kernel_v1 | cublas_gemm_test | cutlass_gemm_test |
| --- | --- | --- | --- |
| M=512, N=512, K=512 | 3.409 | 1.738 | 1.770 |
| M=1024, N=512, K=2048 | 7.217 | 7.857 | 3.699 |
| M=2048, N=2048, K=2048 | 20.718 | 39.993 | 10.404 |
| M=4096, N=4096, K=4096 | 34.765 | 179.772 | 15.149 |
| M=8192, N=4096, K=2048 | 36.745 | 172.096 | 17.305 |
| M=8192, N=8192, K=8192 | 37.701 | 268.805 | 17.390 |

- Speedup

| Problem Size | GEMM_kernel_v1 | cublas_gemm_test | cutlass_gemm_test |
| --- | --- | --- | --- |
| M=512, N=512, K=512 | 2.01x | 1.02x | 1.04x |
| M=1024, N=512, K=2048 | 0.92x | 1.00x | 0.47x |
| M=2048, N=2048, K=2048 | 0.52x | 1.00x | 0.26x |
| M=4096, N=4096, K=4096 | 0.23x | 1.21x | 0.10x |
| M=8192, N=4096, K=2048 | 0.21x | 1.00x | 0.10x |
| M=8192, N=8192, K=8192 | 0.14x | 0.99x | 0.06x |

### V2

The V2 version 

#### Asynchronize Memory Operators

https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-comparison-instructions

`cp.async`: Initiates an asynchronous copy operation from one state space to another.

```asm
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], cp-size{, src-size}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], 16{, src-size}{, cache-policy} ;
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], cp-size{, ignore-src}{, cache-policy} ;
cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}
                         [dst], [src], 16{, ignore-src}{, cache-policy} ;

.level::cache_hint =     { .L2::cache_hint }
.level::prefetch_size =  { .L2::64B, .L2::128B, .L2::256B }
cp-size =                { 4, 8, 16 }
```


The difference between `ca` and `cg` is:

- `ca`: The default cache updating method. Update cache at all levels.
- `cg`: Cache at global level. Use ld.cg to cache loads only globally, bypassing the L1 cache, and cache only in the L2 cache.


## SM89 Normal BF16 GEMM Kernel

### V2



#### Performance in RTX4090


| Problem Size | GEMM_kernel_v1 | GEMM_kernel_v2 | cublas_gemm_test | cutlass_gemm_test | reference_cublas |
| --- | --- | --- | --- | --- | --- |
| M=512, N=512, K=512 | 6.096 | 8.978 | 1.090 | 4.846 | 34.953 |
| M=1024, N=512, K=2048 | 12.780 | 19.222 | 8.405 | 10.195 | 119.156 |
| M=2048, N=2048, K=2048 | 69.442 | 94.733 | 48.335 | 40.692 | 151.968 |
| M=4096, N=4096, K=4096 | 82.241 | 105.975 | 115.575 | 41.218 | 145.636 |
| M=8192, N=4096, K=2048 | 82.055 | 105.792 | 126.968 | 43.025 | 162.826 |

Speedup

| Problem Size | GEMM_kernel_v1 | GEMM_kernel_v2 | cublas_gemm_test | cutlass_gemm_test | reference_cublas |
| --- | --- | --- | --- | --- | --- |
| M=512, N=512, K=512 | 0.17x | 0.26x | 0.03x | 0.14x | 1.00x |
| M=1024, N=512, K=2048 | 0.11x | 0.16x | 0.07x | 0.09x | 1.00x |
| M=2048, N=2048, K=2048 | 0.46x | 0.62x | 0.32x | 0.27x | 1.00x |
| M=4096, N=4096, K=4096 | 0.56x | 0.73x | 0.79x | 0.28x | 1.00x |
| M=8192, N=4096, K=2048 | 0.50x | 0.65x | 0.78x | 0.26x | 1.00x |



### V3

In this step, we add swizzling to 