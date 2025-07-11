This repository implements a mix precision GEMM kernel in 8 bits and 16 bits bitwidth precision. The input matrix will be quantized to 8 bits or 16 bits, and the output matrix will be a higher precision such as 16 bits or 32 bits.

Different from the $C=A\times B$, we implement the following formula:
$$C = A\circ Q_A \times B\circ Q_B$$
, where $Q_A$ and $Q_B$ are quantization coefficients.

The kernels will be written in CUDA C++. And our code structure is based on the [RadeonFlow_Kernels](https://github.com/RadeonFlow/RadeonFlow_Kernels.git).

# CUDA Implementation

- [ ] normal FP16 & BF16 GEMM
- [ ] mixed int8 GEMM

## Normal FP16 & BF16 GEMM

### SM80

To review the GEMM kernel and test our code struture, we first implement an normal FP16 GEMM kernel in the NVIDIA's sm80 architecture. 

#### Optimization

- [ ] Transpose input matrices to improve L2 cache hit rate and HBM bandwidth utilization.
- [ ] Batch GDS-to-LDS memory loads.
- [ ] Employing asynchronous buffer load instructions 
- [ ] Pad shared memory to avoid bank conflicts.
- [ ] Split-K to improve CU occupancy when M and N are small but K is large.
- [ ] Pipeline GDS-to-LDS loading and Matrix Core computation (sm80 special Instruction).
- [ ] Fast (but unsafe) float-to-bfloat16 casting.
- [ ] Block swizzling to further increase L2 cache hit rate.
- [ ] Swizzling data in LDS to avoid explicit padding.

# Triton Implementation

- [ ] normal FP16 & BF16 GEMM
- [ ] mixed int8 GEMM