#include <iostream>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// CUTLASS
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

template <typename typeIn, typename typeOut>
void cutlass_gemm_test(
    typeIn* A, size_t ldA,
    typeIn* B, size_t ldB,
    typeOut* C, size_t ldC,
    size_t M, size_t N, size_t K);

template <>
void cutlass_gemm_test<__nv_bfloat16, float>(
    __nv_bfloat16* A, size_t ldA,
    __nv_bfloat16* B, size_t ldB,
    float* C, size_t ldC,
    size_t M, size_t N, size_t K) {

  using ElementA = __nv_bfloat16;
  using ElementB = __nv_bfloat16;
  using ElementC = float;
  using ElementAccumulator = float;

  using Layout = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementA, Layout,
      ElementB, Layout,
      ElementC, Layout,
      ElementAccumulator>;

  Gemm gemm_op;

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // alpha/beta 用累加计算类型（float）
  typename Gemm::Arguments args{
      problem_size,
      {A, static_cast<int>(ldA)},
      {B, static_cast<int>(ldB)},
      {C, static_cast<int>(ldC)},   // C (source)
      {C, static_cast<int>(ldC)},   // D (destination)
      {1.0f, 0.0f}
  };

  auto status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM (bf16) failed: " << int(status) << std::endl;
  }
}

template <>
void cutlass_gemm_test<int8_t, int8_t>(
    int8_t* A, size_t ldA,
    int8_t* B, size_t ldB,
    int8_t* C, size_t ldC,
    size_t M, size_t N, size_t K) {

  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementC = int8_t;
  using ElementAccumulator = int32_t;

  using Layout = cutlass::layout::RowMajor;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementA, Layout,
      ElementB, Layout,
      ElementC, Layout,
      ElementAccumulator>;

  Gemm gemm_op;

  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // alpha/beta 和累加器一致（int32）
  typename Gemm::Arguments args{
      problem_size,
      {A, static_cast<int>(ldA)},
      {B, static_cast<int>(ldB)},
      {C, static_cast<int>(ldC)},   // C (source)
      {C, static_cast<int>(ldC)},   // D (destination)
      {ElementAccumulator(1), ElementAccumulator(0)}
  };

  auto status = gemm_op(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM (int8) failed: " << int(status) << std::endl;
  }
}