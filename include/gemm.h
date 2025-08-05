/*! \file gemm.h
    \brief API to access matrix multiply-accumulate operations
*/

#pragma once

template <typename typeIn, typename typeOut>
void cutlass_gemm_test(typeIn* A, size_t ldA,
    typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

