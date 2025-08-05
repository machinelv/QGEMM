#include <iostream>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/gemm/device/gemm_configuration.h>

#include "gemm.h"


// Define a wrapper function for CUTLASS GEMM without PyTorch dependency
template <typename typeIn, typename typeOut>
void cutlass_gemm_test(
    typeIn* A, size_t ldA,
    typeIn* B, size_t ldB, 
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K) {
    
    // CUTLASS GEMM configuration
    using ElementInputA = typeIn;
    using ElementInputB = typeIn;
    using ElementOutput = typeOut;
    using ElementAccumulator = typeOut;

    using GEMM = cutlass::gemm::device::Gemm<
        ElementInputA, cutlass::layout::RowMajor,
        ElementInputB, cutlass::layout::RowMajor,
        ElementOutput, cutlass::layout::RowMajor,
        ElementAccumulator>;

    GEMM gemm_op;

    // Define problem size
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Launch CUTLASS GEMM
    auto result = gemm_op({
        problem_size,
        {reinterpret_cast<ElementInputA*>(A), ldA},
        {reinterpret_cast<ElementInputB*>(B), ldB},
        {reinterpret_cast<ElementOutput*>(C), ldC},
        {reinterpret_cast<ElementOutput*>(C), ldC},
        {1.0f, 0.0f}
    });

    if (result != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed" << std::endl;
    }
}
