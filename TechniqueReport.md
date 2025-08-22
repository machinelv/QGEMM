This is a technique report for this GEMM repository, which implements both normal and mixed precision GEMM kernels. The kernels are designed to optimize matrix multiplication operations using CUDA C++ and Triton, with a focus on performance enhancements through various optimization techniques.
 


# GEMM

## SM80 Normal BF16 GEMM Kernel

The SM80 kernel is designed to leverage the capabilities of the NVIDIA Ampere architecture for efficient matrix multiplication. The input matrix precision is bf16 while the output's is fp32.

- V1: tensor core instructions, no swizzling.
- V2: tensor core instructions, asynchronize smem storing.
- V3: tensor core instructions, asynchronize smem storing, swizzling.
- V4: tensor core instructions, asynchronize smem storing, swizzling, split-K
- V5: tensor core instructions, asynchronize smem storing, swizzling, split-K, pipeline-K


### V1 

The V1 is a simple version of GEMM. It uses the techniques below:
- Tensor core instructions
- Vectorized Memory Fetching

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

