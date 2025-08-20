/*! \file gemm.h
    \brief API to access matrix multiply-accumulate operations
*/

#pragma once
#include "common.h"


#ifdef USE_CUDA

template <typename typeIn, typename typeOut>
void cutlass_gemm_test(typeIn* A, size_t ldA,
    typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

template<typename typeIn, typename typeOut>
void GEMM_kernel_v1(const typeIn* A, size_t ldA,
    const typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

// Template specializations
template<>
void GEMM_kernel_v1<__nv_bfloat16, __nv_bfloat16>(const __nv_bfloat16* A, size_t ldA,
    const __nv_bfloat16* B, size_t ldB, 
    __nv_bfloat16* C, size_t ldC,
    size_t M, size_t N, size_t K);

#endif