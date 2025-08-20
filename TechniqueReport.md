This is a technique report for this GEMM repository, which implements both normal and mixed precision GEMM kernels. The kernels are designed to optimize matrix multiplication operations using CUDA C++ and Triton, with a focus on performance enhancements through various optimization techniques.
 


# GEMM

## SM80 Normal BF16 GEMM Kernel

- V1: tensor core instructions, no swizzling.
- V2: tensor core instructions, asynchronize smem storing.
- V3: tensor core instructions, asynchronize smem storing, swizzling.
- V4: tensor core instructions, asynchronize smem storing, swizzling, split-K
- V5: tensor core instructions, asynchronize smem storing, swizzling, split-K, pipeline-K



