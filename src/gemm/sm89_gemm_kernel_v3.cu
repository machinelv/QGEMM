

#include <iostream>
#include <cstdlib>

#include "common.h"
#include "gpu_lib.h"
#include "gpu_type.h"
#include "fast_quat.h"
#include "gemm_utils.cuh"
#include "gemm_config.h"


#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

using namespace gemm_kernel;
using WMMA_BF16_Config = gemm_kernel::sm89::WMMA_BF16_Config;

namespace {
    template <  size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
            size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N, 
            size_t WMMA_TILE_PER_WARP_M, size_t WMMA_TILE_PER_WARP_N, size_t WMMA_TILE_PER_WARP_K,
            size_t WMMA_TILE_SIZE_M, size_t WMMA_TILE_SIZE_N, size_t WMMA_TILE_SIZE_K,
            size_t WMMA_REG_PER_THREAD_A, size_t WMMA_REG_PER_THREAD_B, size_t WMMA_REG_PER_THREAD_C>
inline __device__ void process_data_from_shared_memory_using_wmma_bf16_swizzle(
        const __nv_bfloat16* A_shared_block_tile,
        const __nv_bfloat16* B_shared_block_tile,
        uint32_t a_frag[WMMA_TILE_PER_WARP_M][WMMA_REG_PER_THREAD_A],
        uint32_t b_frag[WMMA_TILE_PER_WARP_N][WMMA_REG_PER_THREAD_B],
        float acc_frag_fp32[WMMA_TILE_PER_WARP_M][WMMA_TILE_PER_WARP_N][WMMA_REG_PER_THREAD_C],
        size_t M, size_t N, size_t K,
        size_t warp_m_id, size_t warp_n_id)
{
    static_assert(WMMA_REG_PER_THREAD_A == 4);
    static_assert(WMMA_REG_PER_THREAD_B == 4);
    static_assert(WMMA_REG_PER_THREAD_C == 4);

    size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    size_t lane_id = thread_id % WARP_SIZE;

    // process wmma tile
    #pragma unroll(WMMA_TILE_PER_WARP_K)
    for (size_t wmma_tile_idx_k{0U}; wmma_tile_idx_k < WMMA_TILE_PER_WARP_K; ++wmma_tile_idx_k) {
        // Load data from shared memory to register
        size_t block_tile_wmma_tile_k_idx{wmma_tile_idx_k * WMMA_TILE_SIZE_K};
        // Load A and B matrices from shared memory to registers
        #pragma unroll(WMMA_TILE_PER_WARP_M)
        for (size_t wmma_tile_idx_m{0U}; wmma_tile_idx_m < WMMA_TILE_PER_WARP_M; ++wmma_tile_idx_m) {
            size_t block_tile_wmma_tile_m_idx{warp_m_id * WARP_TILE_SIZE_M + wmma_tile_idx_m * WMMA_TILE_SIZE_M};
            // wmma::load_matrix_sync(a_frag[wmma_tile_idx_m], &A_shared_block_tile[block_tile_wmma_tile_k_idx + block_tile_wmma_tile_m_idx * BLOCK_TILE_SIZE_K], BLOCK_TILE_SIZE_K);
            size_t row_id, col_id;
            shared_memory_swizzle_coordinate<3,3,3,__nv_bfloat16>(lane_id, row_id, col_id);
            uint32_t smem_addr = __cvta_generic_to_shared(&A_shared_block_tile[row_id * BLOCK_TILE_SIZE_K + col_id]);
            LDMATRIX_BF16_X4(a_frag[wmma_tile_idx_m][0], a_frag[wmma_tile_idx_m][1], a_frag[wmma_tile_idx_m][2], a_frag[wmma_tile_idx_m][3], smem_addr);
        }
        #pragma unroll(WMMA_TILE_PER_WARP_N)
        for (size_t wmma_tile_idx_n{0U}; wmma_tile_idx_n < WMMA_TILE_PER_WARP_N; ++wmma_tile_idx_n) {
            size_t block_tile_wmma_tile_n_idx{warp_n_id * WARP_TILE_SIZE_N + wmma_tile_idx_n * WMMA_TILE_SIZE_N};
            // wmma::load_matrix_sync(b_frag[wmma_tile_idx_n], &B_shared_block_tile[block_tile_wmma_tile_k_idx * BLOCK_TILE_SIZE_N + block_tile_wmma_tile_n_idx], BLOCK_TILE_SIZE_N);
            size_t row_id, col_id;
            shared_memory_swizzle_coordinate<3,3,3,__nv_bfloat16>(lane_id, &row_id, &col_id);
            uint32_t smem_addr = __cvta_generic_to_shared(&B_shared_block_tile[row_id * BLOCK_TILE_SIZE_N + col_id]);
            LDMATRIX_BF16_TRANSPOSE_X4(b_frag[wmma_tile_idx_n][0], b_frag[wmma_tile_idx_n][1], b_frag[wmma_tile_idx_n][2], b_frag[wmma_tile_idx_n][3], smem_addr);
        }
        // Compute the acc_frag_fp32
        #pragma unroll(WMMA_TILE_PER_WARP_M)
        for (size_t wmma_tile_idx_m{0U}; wmma_tile_idx_m < WMMA_TILE_PER_WARP_M; ++wmma_tile_idx_m) {
            #pragma unroll(WMMA_TILE_PER_WARP_N)
            for (size_t wmma_tile_idx_n{0U}; wmma_tile_idx_n < WMMA_TILE_PER_WARP_N; ++wmma_tile_idx_n) {
                // wmma::mma_sync(acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n], a_frag[wmma_tile_idx_m], b_frag[wmma_tile_idx_n], acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n]);
                BF16MMA16816(acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n][0], acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n][1],
                a_frag[wmma_tile_idx_m][0], a_frag[wmma_tile_idx_m][1], a_frag[wmma_tile_idx_m][2], a_frag[wmma_tile_idx_m][3],
                b_frag[wmma_tile_idx_n][0], b_frag[wmma_tile_idx_n][1],
                acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n][0], acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n][1]);

                BF16MMA16816(acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n][2], acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n][3],
                a_frag[wmma_tile_idx_m][0], a_frag[wmma_tile_idx_m][1], a_frag[wmma_tile_idx_m][2], a_frag[wmma_tile_idx_m][3],
                b_frag[wmma_tile_idx_n][2], b_frag[wmma_tile_idx_n][3],
                acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n][2], acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n][3]);
            }
        }
    }
    __syncthreads();
}
}


namespace gemm_kernel{
namespace sm89{


template <typename T_OUTPUT, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
          size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N, 
          size_t WMMA_TILE_SIZE_M, size_t WMMA_TILE_SIZE_N, size_t WMMA_TILE_SIZE_K,
          size_t GROUP_SIZE_M>
__global__ void GEMM_kernel_bf16_v3(const __nv_bfloat16* A, size_t ldA, const __nv_bfloat16* B, size_t ldB,
                       float* C, size_t ldC, size_t M, size_t N, size_t K)
{
    static_assert(BLOCK_TILE_SIZE_M % WARP_TILE_SIZE_M == 0, 
                  "BLOCK_TILE_SIZE_M must be divisible by WARP_TILE_SIZE_M");
    static_assert(BLOCK_TILE_SIZE_N % WARP_TILE_SIZE_N == 0,
                    "BLOCK_TILE_SIZE_N must be divisible by WARP_TILE_SIZE_N");
    static_assert(BLOCK_TILE_SIZE_K % WMMA_TILE_SIZE_K == 0,   
                    "BLOCK_TILE_SIZE_M must be divisible by WMMA_TILE_SIZE_M");

    constexpr size_t WARP_NUM_M{BLOCK_TILE_SIZE_M / WARP_TILE_SIZE_M};
    constexpr size_t WARP_NUM_N{BLOCK_TILE_SIZE_N / WARP_TILE_SIZE_N};
    constexpr size_t WARP_NUM{WARP_NUM_M * WARP_NUM_N};
    constexpr size_t THREAD_NUM{WARP_NUM * WARP_SIZE};

    constexpr size_t WMMA_TILE_PER_WARP_M{WARP_TILE_SIZE_M / WMMA_TILE_SIZE_M};
    constexpr size_t WMMA_TILE_PER_WARP_N{WARP_TILE_SIZE_N / WMMA_TILE_SIZE_N};
    constexpr size_t WMMA_TILE_PER_WARP_K{BLOCK_TILE_SIZE_K / WMMA_TILE_SIZE_K};

    constexpr size_t WMMA_REG_PER_THREAD_A{WMMA_TILE_SIZE_M * WMMA_TILE_SIZE_K / WARP_NUM / sizeof(uint32_t)};
    constexpr size_t WMMA_REG_PER_THREAD_B{WMMA_TILE_SIZE_N * WMMA_TILE_SIZE_K / WARP_NUM / sizeof(uint32_t)};
    constexpr size_t WMMA_REG_PER_THREAD_C{WMMA_TILE_SIZE_N * WMMA_TILE_SIZE_M / WARP_NUM / sizeof(uint32_t)};

    // Calculate the block row and block column indices
    size_t block_id = blockIdx.x + blockIdx.y * gridDim.x;
    size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    size_t warp_id = thread_id / WARP_SIZE;
    size_t warp_m_id = warp_id / WARP_NUM_N; // warp_row_idx
    size_t warp_n_id = warp_id % WARP_NUM_N; // warp_col_idx

    // size_t block_tile_num_m = (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M;
    size_t block_tile_num_n = (N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N;

    size_t block_tile_start_m, block_tile_start_n;

    if constexpr (GROUP_SIZE_M > 1) {
        // swizzle the block index to improve L2 cache locality
        size_t group_block_num = GROUP_SIZE_M * block_tile_num_n;
        size_t group_id = block_id / group_block_num;
        size_t group_block_start_m = group_id * GROUP_SIZE_M;
        size_t group_block_size_m = min(GROUP_SIZE_M, M - group_block_start_m);

        size_t block_tile_id_m = group_block_start_m + ((block_id % group_block_num) % group_block_size_m);
        size_t block_tile_id_n = (block_id % group_block_num) / group_block_size_m;
        block_tile_start_m = block_tile_id_m * BLOCK_TILE_SIZE_M;
        block_tile_start_n = block_tile_id_n * BLOCK_TILE_SIZE_N;
    } else {
        block_tile_start_m = blockIdx.y * BLOCK_TILE_SIZE_M;
        block_tile_start_n = blockIdx.x * BLOCK_TILE_SIZE_N;
    }

    // __shared__ __nv_bfloat16 A_block_tile[BLOCK_TILE_SIZE_M][BLOCK_TILE_SIZE_K];
    // __shared__ __nv_bfloat16 B_shared_memory[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];
    // __nv_bfloat16* A_block_tile = &A_block_tile[0][0];
    // __nv_bfloat16* B_block_tile = &B_shared_memory[0][0];

    extern __shared__ __nv_bfloat16 shared_memory[];
    __nv_bfloat16* A_block_tile = &shared_memory[0];
    __nv_bfloat16* B_block_tile = &shared_memory[BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_M];

    // wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, float> acc_frag_fp32[WMMA_TILE_PER_WARP_M][WMMA_TILE_PER_WARP_N];
    uint32_t acc_frag_fp32[WMMA_TILE_PER_WARP_M][WMMA_TILE_PER_WARP_N][WMMA_REG_PER_THREAD_C];

    #pragma unroll(WMMA_TILE_PER_WARP_M)
    for (size_t wmma_tile_m_idx{0U}; wmma_tile_m_idx < WMMA_TILE_PER_WARP_M; ++wmma_tile_m_idx){
        #pragma unroll(WMMA_TILE_PER_WARP_N)
        for (size_t wmma_tile_n_idx{0U}; wmma_tile_n_idx < WMMA_TILE_PER_WARP_N;++wmma_tile_n_idx) {
            #pragma unroll(WMMA_REG_PER_THREAD_C)
            for (size_t reg_idx{0U}; reg_idx < WMMA_REG_PER_THREAD_C; ++reg_idx) {
                acc_frag_fp32[wmma_tile_m_idx][wmma_tile_n_idx][reg_idx] = static_cast<uint32_t>(0);
            }
        }
    }
    
    for (size_t block_id_k{0}; block_id_k < K; block_id_k += BLOCK_TILE_SIZE_K)
    {
        // wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, __nv_bfloat16, wmma::row_major> a_frag[WMMA_TILE_PER_WARP_M];
        // wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, __nv_bfloat16, wmma::row_major> b_frag[WMMA_TILE_PER_WARP_N];

        uint32_t a_frag[WMMA_TILE_PER_WARP_M][WMMA_REG_PER_THREAD_A];
        uint32_t b_frag[WMMA_TILE_PER_WARP_N][WMMA_REG_PER_THREAD_B];

        size_t block_tile_start_k = block_id_k;
        // load A and B matrices from global memory to shared memory
        load_data_from_global_memory_to_shared_memory_async
            <__nv_bfloat16, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K,
            WARP_TILE_SIZE_M, WARP_TILE_SIZE_N, THREAD_NUM>
            (A, B, A_block_tile, B_block_tile, M, N, K,
             block_tile_start_m, block_tile_start_n, block_tile_start_k, thread_id);
        cp_async_wait_all;
        __syncthreads();

        // load a_frag and b_frag from shared memory to registers and compute
        process_data_from_shared_memory_using_wmma_bf16_swizzle<
            BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K,
            WARP_TILE_SIZE_M, WARP_TILE_SIZE_N, 
            WMMA_TILE_PER_WARP_M, WMMA_TILE_PER_WARP_N, WMMA_TILE_PER_WARP_K,
            WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K>
            (A_block_tile, B_block_tile, a_frag, b_frag, acc_frag_fp32, M, N, K, warp_m_id, warp_n_id);
        __syncthreads();
    }


    // Store the result to global memory
    #pragma unroll(WMMA_TILE_PER_WARP_M)
    for (size_t wmma_tile_idx_m{0U}; wmma_tile_idx_m < WMMA_TILE_PER_WARP_M; ++wmma_tile_idx_m) {
        #pragma unroll(WMMA_TILE_PER_WARP_N)
        for (size_t wmma_tile_idx_n{0U}; wmma_tile_idx_n < WMMA_TILE_PER_WARP_N; ++wmma_tile_idx_n) {
            size_t M_idx = block_tile_start_m + warp_m_id * WARP_TILE_SIZE_M + wmma_tile_idx_m * WMMA_TILE_SIZE_M;
            size_t N_idx = block_tile_start_n + warp_n_id * WARP_TILE_SIZE_N + wmma_tile_idx_n * WMMA_TILE_SIZE_N;
            if (M_idx < M && N_idx < N) {
                wmma::store_matrix_sync(
                    &C[M_idx * ldC + N_idx], acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n], ldC, wmma::mem_row_major);
            }
        }
    }
}

/**
 * A normal matrix multiplication kernel
 * A, B are input matrices, C is the output matrix.
 * All matrix are stored in row-major order.
 */
template<size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K, size_t GROUP_SIZE_M>
void launch_bf16_kernel_v3(const __nv_bfloat16* A, size_t ldA, const __nv_bfloat16* B, size_t ldB,
                       float* C, size_t ldC, size_t M, size_t N, size_t K) 
{
    constexpr size_t WARP_NUM_M{BLOCK_TILE_SIZE_M / WMMA_BF16_Config::WARP_TILE_SIZE_M};
    constexpr size_t WARP_NUM_N{BLOCK_TILE_SIZE_N / WMMA_BF16_Config::WARP_TILE_SIZE_N};
    constexpr size_t WARP_NUM{WARP_NUM_M * WARP_NUM_N};
    constexpr size_t THREAD_NUM{WARP_NUM * WARP_SIZE};

    GEMM_kernel_bf16_v3<float, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, 
                    WMMA_BF16_Config::WARP_TILE_SIZE_M, WMMA_BF16_Config::WARP_TILE_SIZE_N, 
                    WMMA_BF16_Config::WMMA_TILE_SIZE_M, WMMA_BF16_Config::WMMA_TILE_SIZE_N, WMMA_BF16_Config::WMMA_TILE_SIZE_K, 
                    GROUP_SIZE_M>
        <<<dim3((N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N, (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M, 1),
           dim3(THREAD_NUM), BLOCK_TILE_SIZE_K*(BLOCK_TILE_SIZE_N + BLOCK_TILE_SIZE_M)*sizeof(__nv_bfloat16)>>>(A, ldA, B, ldB, C, ldC, M, N, K);
}
}           // namespace sm89
}           // namespace gemm_kernel

#if CUDA_ARCH == 89
template<typename typeIn, typename typeOut>
void GEMM_kernel_v3(const typeIn* A, size_t ldA,
                    const typeIn* B, size_t ldB, 
                    typeOut* C, size_t ldC,
                    size_t M, size_t N, size_t K);

template<>
void GEMM_kernel_v3<__nv_bfloat16, float>(const __nv_bfloat16* A, size_t ldA,
                    const __nv_bfloat16* B, size_t ldB, 
                    float* C, size_t ldC,
                    size_t M, size_t N, size_t K)
{
    // Implementation for __nv_bfloat16 - call launch function with constants
    constexpr GEMM_Config config = sm89::config_128_128_64_1_4;
    sm89::launch_bf16_kernel_v3<config.BLOCK_TILE_SIZE_M, config.BLOCK_TILE_SIZE_N, config.BLOCK_TILE_SIZE_K, config.GROUP_SIZE_M>(A, ldA, B, ldB, C, ldC, M, N, K);
}

// Explicit template instantiation
template void GEMM_kernel_v3<__nv_bfloat16, float>(const __nv_bfloat16*, size_t, const __nv_bfloat16*, size_t, float*, size_t, size_t, size_t, size_t);

#ifdef LOCAL_TEST

int main() {

    const size_t M = 1024;
    const size_t N = 1024;
    const size_t K = 1024;

    __nv_bfloat16* A;
    __nv_bfloat16* B;
    float* C;

    cudaMalloc(&A, M * K * sizeof(__nv_bfloat16));
    cudaMalloc(&B, K * N * sizeof(__nv_bfloat16));
    cudaMalloc(&C, M * N * sizeof(float));

    // Initialize A and B with some values

    GEMM_kernel_v3(A, K, B, N, C, N, M, N, K);

    // Cleanup
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}

#endif

#endif