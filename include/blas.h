#pragma once



#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
using blas_handle_t = cublasHandle_t;

#endif

#ifdef USE_HIP 
#include <rocblas/rocblas.h>
using blas_handle_t = rocblas_handle;
#endif

#if !defined(USE_CUDA) && !defined(USE_HIP)
#error "Either USE_CUDA or USE_HIP must be defined"
#endif



void blas_init(blas_handle_t &handle) {
#ifdef USE_CUDA
    cublasCreate(&handle);
#endif

#ifdef USE_HIP
    rocblas_create_handle(&handle);
#endif
}

void blas_destroy(blas_handle_t &handle) {
#ifdef USE_CUDA
    cublasDestroy(handle);
#endif 
#ifdef USE_HIP  
    rocblas_destroy_handle(handle);
#endif
}

void blas_hgemm(blas_handle_t &handle,
                const char transa,
                const char transb,
                size_t m,
                size_t n,
                size_t k,
                const __half *alpha,
                const __half *A,
                size_t lda,
                const __half *B,
                size_t ldb,
                const __half *beta,
                __half *C,
                size_t ldc) {
#ifdef USE_CUDA
    cublasHgemm(handle, transa, transb, m, n, k, &alpha,
                    A, ldb, B, lda, &beta, C, ldc);
#endif

#ifdef USE_HIP
    rocblas_hgemm(handle, transa, transb, m, n, k, &alpha,
                    A, ldb, B, lda, &beta, C, ldc);
#endif
}


void blas_bf16gemm(blas_handle_t &handle,
                   const char transa,
                    const char transb,
                    size_t m,
                    size_t n,
                    size_t k,
                    const __half *alpha,
                    const __half *A,
                    size_t lda,
                    const __half *B,
                    size_t ldb,
                    const __half *beta,
                    __half *C,
                    size_t ldc) {
#ifdef USE_CUDA
    
#endif

}

