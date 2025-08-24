#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <mma.h>

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

namespace wmma = nvcuda::wmma;

#define LDMATRIX_BF16_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define LDMATRIX_BF16_TRANSPOSE_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define BF16F32MMA16816(Rd0, Rd1, Rd2, Rd3, Ra0, Ra1, Ra2, Ra3, Rb0, Rb1, Rc0, Rc1, Rc2, Rc3)         \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32         \
                    {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"      \
                 : "=r"(Rd0), "=r"(Rd1), "=r"(Rd2), "=r"(Rd3)                                     \
                 : "r"(Ra0), "r"(Ra1), "r"(Ra2), "r"(Ra3),                  \
                   "r"(Rb0), "r"(Rb1),                                      \
                   "r"(Rc0), "r"(Rc1), "r"(Rc2), "r"(Rc3))

#define copy_cg_async(smem_ptr, gmem_ptr, BTYES)                \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"  \
                :: "r"(smem_ptr),                               \
                "l"(gmem_ptr),                                  \
                "n"(BTYES));

#define copy_async_commit                           \
    asm volatile("cp.async.commit_group;\n" ::);

#define cp_async_wait(N)                                    \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));

#define  cp_async_wait_all      \
    asm volatile("cp.async.wait_all;\n"  ::);

#define LIB_CALL(call)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                           \
    if (err != cudaSuccess) {                                                                                          \
      abort();                                                                                                         \
    }                                                                                                                  \
  } while (0)

#define HOST_TYPE(x) cuda##x

#else

#ifndef HIP_HEADERS__
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#define HIP_HEADERS__
#endif

namespace wmma = rocwmma;

#define LIB_CALL(call)                                                                                                 \
  do {                                                                                                                 \
    hipError_t err = call;                                                                                             \
    if (err != hipSuccess) {                                                                                           \
      abort();                                                                                                         \
    }                                                                                                                  \
  } while (0)

#define HOST_TYPE(x) hip##x

#endif

using wmma::matrix_a;
using wmma::matrix_b;
using wmma::accumulator;
using wmma::col_major;
using wmma::row_major;

#define make_uint32(a) (((uint32_t*)(&(a)))[0])