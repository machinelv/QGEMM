
#include "blas.h"


template <typename typeIn, typename typeOut>
void cublas_gemm_test(
    typeIn* A, size_t ldA,
    typeIn* B, size_t ldB,
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

