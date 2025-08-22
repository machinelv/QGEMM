
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

template <typename typeIn, typename typeOut>
void cublas_gemm_test(
    typeIn* A, size_t ldA,
    typeIn* B, size_t ldB,
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);


template<>
void cublas_gemm_test<__nv_bfloat16, __nv_bfloat16>(
    __nv_bfloat16* A, size_t ldA,
    __nv_bfloat16* B, size_t ldB,
    __nv_bfloat16* C, size_t ldC,
    size_t M, size_t N, size_t K) {

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS create failed: " << status << std::endl;
    return;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  status = cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               N, M, K,
               &alpha,
               A, CUDA_R_16BF, ldA,
               B, CUDA_R_16BF, ldB,
               &beta,
               C, CUDA_R_16BF, ldC,
               CUDA_R_32F,
               CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS GEMM (bf16->bf16) failed: " << status << std::endl;
  }

  cublasDestroy(handle);
}

template<>
void cublas_gemm_test<__nv_bfloat16, float>(
    __nv_bfloat16* A, size_t ldA,
    __nv_bfloat16* B, size_t ldB,
    float* C, size_t ldC,
    size_t M, size_t N, size_t K) {

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS create failed: " << status << std::endl;
    return;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  status = cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               N, M, K,
               &alpha,
               B, CUDA_R_16BF, ldB,
               A, CUDA_R_16BF, ldA,
               &beta,
               C, CUDA_R_32F, ldC,
               CUDA_R_32F,
               CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "ldA=" << ldA << ", ldB=" << ldB << ", ldC=" << ldC << std::endl;
    std::cerr << "cuBLAS GEMM (bf16->float) failed: " << status << std::endl;
  }

  cublasDestroy(handle);
}



template<>
void cublas_gemm_test<int8_t, int8_t>(
    int8_t* A, size_t ldA,
    int8_t* B, size_t ldB,
    int8_t* C, size_t ldC,
    size_t M, size_t N, size_t K) {

  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS create failed: " << status << std::endl;
    return;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  status = cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               N, M, K,
               &alpha,
               A, CUDA_R_8I, ldA,
               B, CUDA_R_8I, ldB,
               &beta,
               C, CUDA_R_8I, ldC,
               CUDA_R_32F,
               CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS GEMM (int8) failed: " << status << std::endl;
  }

  cublasDestroy(handle);
}