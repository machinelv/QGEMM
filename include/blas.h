#pragma once



#ifdef TEST_ON_CUDA
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
using blas_handle_t = cublasHandle_t;

#endif

#ifdef TEST_ON_HIP 
#include <rocblas/rocblas.h>
using blas_handle_t = rocblas_handle;
#endif



void blas_init(blas_handle_t &handle) {
#ifdef TEST_ON_CUDA
    cublasCreate(&handle);
#endif

#ifdef TEST_ON_HIP
    rocblas_create_handle(&handle);
#endif
}

void blas_destroy(blas_handle_t &handle) {
#ifdef TEST_ON_CUDA
    cublasDestroy(handle);
#endif 
#ifdef TEST_ON_HIP  
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
#ifdef TEST_ON_CUDA
    cublasHgemm(handle, transa, transb, m, n, k, &alpha,
                    A, ldb, B, lda, &beta, C, ldc);
#endif

#ifdef TEST_ON_HIP
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
#ifdef TEST_ON_CUDA
    
#endif

}


