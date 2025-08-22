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

namespace blas{

blas_handle_t blas_handle;

void blas_init(blas_handle_t &handle) {
#ifdef USE_CUDA
    cublasCreate(&handle);
#endif

#ifdef USE_HIP
    rocblas_create_handle(&handle);
#endif
}

void blas_init() {
    blas_init(blas_handle);
}

void blas_destroy(blas_handle_t &handle) {
#ifdef USE_CUDA
    cublasDestroy(handle);
#endif 
#ifdef USE_HIP  
    rocblas_destroy_handle(handle);
#endif
}

void blas_destroy() {
    blas_destroy(blas_handle);
}

}