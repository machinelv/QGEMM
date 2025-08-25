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

template <typename typeIn, typename typeOut>
void cublas_gemm_test(
    typeIn* A, size_t ldA,
    typeIn* B, size_t ldB,
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);


template<typename typeIn, typename typeOut>
void GEMM_kernel_v1(const typeIn* A, size_t ldA,
    const typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);


template<typename typeIn, typename typeOut>
void GEMM_kernel_v2(const typeIn* A, size_t ldA,
    const typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

template<typename typeIn, typename typeOut>
void GEMM_kernel_v3(const typeIn* A, size_t ldA,
    const typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

template<typename typeIn, typename typeOut>
void GEMM_kernel_v3_1(const typeIn* A, size_t ldA,
    const typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

template<typename typeIn, typename typeOut>
void GEMM_kernel_v4(const typeIn* A, size_t ldA,
    const typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

template<typename typeIn, typename typeOut>
void GEMM_kernel_v4_1(const typeIn* A, size_t ldA,
    const typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

template<typename typeIn, typename typeOut>
void GEMM_kernel_v5(const typeIn* A, size_t ldA,
    const typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

#endif