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
using WMMA_BF16_Config = gemm_kernel::sm80::WMMA_BF16_Config;

namespace gemm_kernel{
namespace sm80{

template <  size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
            size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N, 
            size_t WMMA_TILE_PER_WARP_M, size_t WMMA_TILE_PER_WARP_N, size_t WMMA_TILE_PER_WARP_K,
            size_t WMMA_TILE_SIZE_M, size_t WMMA_TILE_SIZE_N, size_t WMMA_TILE_SIZE_K>
inline __device__ void process_data_from_shared_memory_using_wmma_bf16(
        const __nv_bfloat16* A_T_shared_block_tile,
        const __nv_bfloat16* B_shared_block_tile,
        wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, __nv_bfloat16, wmma::col_major> 
        a_frag[WMMA_TILE_PER_WARP_M],
        wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, __nv_bfloat16, wmma::row_major>
        b_frag[WMMA_TILE_PER_WARP_N],
        wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, float> 
        acc_frag_fp32[WMMA_TILE_PER_WARP_M][WMMA_TILE_PER_WARP_N],
        size_t M, size_t N, size_t K,
        size_t warp_m_id, size_t warp_n_id )
{

    // process wmma tile
    #pragma unroll(WMMA_TILE_PER_WARP_K)
    for (size_t wmma_tile_idx_k{0U}; wmma_tile_idx_k < WMMA_TILE_PER_WARP_K; ++wmma_tile_idx_k) {
        // Load data from shared memory to register
        size_t block_tile_wmma_tile_k_idx{wmma_tile_idx_k * WMMA_TILE_SIZE_K};
        // Load A and B matrices from shared memory to registers
        #pragma unroll(WMMA_TILE_PER_WARP_M)
        for (size_t wmma_tile_idx_m{0U}; wmma_tile_idx_m < WMMA_TILE_PER_WARP_M; ++wmma_tile_idx_m) {
            size_t block_tile_wmma_tile_m_idx{warp_m_id * WARP_TILE_SIZE_M + wmma_tile_idx_m * WMMA_TILE_SIZE_M};
            wmma::load_matrix_sync(a_frag[wmma_tile_idx_m], &A_T_shared_block_tile[block_tile_wmma_tile_k_idx * BLOCK_TILE_SIZE_M + block_tile_wmma_tile_m_idx], BLOCK_TILE_SIZE_M);
        }
        #pragma unroll(WMMA_TILE_PER_WARP_N)
        for (size_t wmma_tile_idx_n{0U}; wmma_tile_idx_n < WMMA_TILE_PER_WARP_N; ++wmma_tile_idx_n) {
            size_t block_tile_wmma_tile_n_idx{warp_n_id * WARP_TILE_SIZE_N + wmma_tile_idx_n * WMMA_TILE_SIZE_N};
            wmma::load_matrix_sync(b_frag[wmma_tile_idx_n], &B_shared_block_tile[block_tile_wmma_tile_k_idx * BLOCK_TILE_SIZE_N + block_tile_wmma_tile_n_idx], BLOCK_TILE_SIZE_N);
        }

        // Compute the acc_frag_fp32
        #pragma unroll(WMMA_TILE_PER_WARP_M)
        for (size_t wmma_tile_idx_m{0U}; wmma_tile_idx_m < WMMA_TILE_PER_WARP_M; ++wmma_tile_idx_m) {
            #pragma unroll(WMMA_TILE_PER_WARP_N)
            for (size_t wmma_tile_idx_n{0U}; wmma_tile_idx_n < WMMA_TILE_PER_WARP_N; ++wmma_tile_idx_n) {
                wmma::mma_sync(acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n], a_frag[wmma_tile_idx_m], b_frag[wmma_tile_idx_n], acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n]);
            }
        }
    }
    __syncthreads();
}


template <typename T_OUTPUT, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
          size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N, 
          size_t WMMA_TILE_SIZE_M, size_t WMMA_TILE_SIZE_N, size_t WMMA_TILE_SIZE_K,
          size_t GROUP_SIZE_M>
__global__ void GEMM_kernel_bf16(const __nv_bfloat16* A, size_t ldA, const __nv_bfloat16* B, size_t ldB,
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

    // __shared__ __nv_bfloat16 A_T_shared_memory[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M];
    // __shared__ __nv_bfloat16 B_shared_memory[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];
    // __nv_bfloat16* A_T_block_tile = &A_T_shared_memory[0][0];
    // __nv_bfloat16* B_block_tile = &B_shared_memory[0][0];

    extern __shared__ __nv_bfloat16 shared_memory[];
    __nv_bfloat16* A_T_block_tile = &shared_memory[0];
    __nv_bfloat16* B_block_tile = &shared_memory[BLOCK_TILE_SIZE_K * BLOCK_TILE_SIZE_M];

    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, float> acc_frag_fp32[WMMA_TILE_PER_WARP_M][WMMA_TILE_PER_WARP_N];
    #pragma unroll(WMMA_TILE_PER_WARP_M)
    for (size_t wmma_tile_m_idx{0U}; wmma_tile_m_idx < WMMA_TILE_PER_WARP_M; ++wmma_tile_m_idx){
        #pragma unroll(WMMA_TILE_PER_WARP_N)
        for (size_t wmma_tile_n_idx{0U}; wmma_tile_n_idx < WMMA_TILE_PER_WARP_N;++wmma_tile_n_idx) {
            wmma::fill_fragment(acc_frag_fp32[wmma_tile_m_idx][wmma_tile_n_idx], static_cast<float>(0));
        }
    }
    
    for (size_t block_id_k{0}; block_id_k < K; block_id_k += BLOCK_TILE_SIZE_K)
    {
        wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, __nv_bfloat16, wmma::col_major> a_frag[WMMA_TILE_PER_WARP_M];
        wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, __nv_bfloat16, wmma::row_major> b_frag[WMMA_TILE_PER_WARP_N];
        size_t block_tile_start_k = block_id_k;
        // load A and B matrices from global memory to shared memory
        load_data_from_global_memory_to_shared_memory_transposed
            <__nv_bfloat16, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, THREAD_NUM>
            (A, B, A_T_block_tile, B_block_tile, M, N, K,
             block_tile_start_m, block_tile_start_n, block_tile_start_k, thread_id);
        __syncthreads();
        
        // load a_frag and b_frag from shared memory to registers and compute
        process_data_from_shared_memory_using_wmma_bf16<
            BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K,
            WARP_TILE_SIZE_M, WARP_TILE_SIZE_N, 
            WMMA_TILE_PER_WARP_M, WMMA_TILE_PER_WARP_N, WMMA_TILE_PER_WARP_K,
            WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K>
            (A_T_block_tile, B_block_tile, a_frag, b_frag, acc_frag_fp32, M, N, K, warp_m_id, warp_n_id);
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
void launch_bf16_kernel_v1(const __nv_bfloat16* A, size_t ldA, const __nv_bfloat16* B, size_t ldB,
                       float* C, size_t ldC, size_t M, size_t N, size_t K) 
{
    constexpr size_t WARP_NUM_M{BLOCK_TILE_SIZE_M / WMMA_BF16_Config::WARP_TILE_SIZE_M};
    constexpr size_t WARP_NUM_N{BLOCK_TILE_SIZE_N / WMMA_BF16_Config::WARP_TILE_SIZE_N};
    constexpr size_t WARP_NUM{WARP_NUM_M * WARP_NUM_N};
    constexpr size_t THREAD_NUM{WARP_NUM * WARP_SIZE};

    GEMM_kernel_bf16<float, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, 
                    WMMA_BF16_Config::WARP_TILE_SIZE_M, WMMA_BF16_Config::WARP_TILE_SIZE_N, 
                    WMMA_BF16_Config::WMMA_TILE_SIZE_M, WMMA_BF16_Config::WMMA_TILE_SIZE_N, WMMA_BF16_Config::WMMA_TILE_SIZE_K, 
                    GROUP_SIZE_M>
        <<<dim3((N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N, (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M, 1),
           dim3(THREAD_NUM), BLOCK_TILE_SIZE_K*(BLOCK_TILE_SIZE_N + BLOCK_TILE_SIZE_M)*sizeof(__nv_bfloat16)>>>(A, ldA, B, ldB, C, ldC, M, N, K);
}
}           // namespace sm80
}           // namespace gemm_kernel

#if CUDA_ARCH == 80
template<typename typeIn, typename typeOut>
void GEMM_kernel_v1(const typeIn* A, size_t ldA,
                    const typeIn* B, size_t ldB, 
                    typeOut* C, size_t ldC,
                    size_t M, size_t N, size_t K);

template<>
void GEMM_kernel_v1<__nv_bfloat16, float>(const __nv_bfloat16* A, size_t ldA,
                    const __nv_bfloat16* B, size_t ldB, 
                    float* C, size_t ldC,
                    size_t M, size_t N, size_t K)
{
    // Implementation for __nv_bfloat16 - call launch function with constants
    sm80::launch_bf16_kernel_v1<sm80::config_128_128_64_1_1.BLOCK_TILE_SIZE_M, sm80::config_128_128_64_1_1.BLOCK_TILE_SIZE_N, sm80::config_128_128_64_1_1.BLOCK_TILE_SIZE_K, sm80::config_128_128_64_1_1.GROUP_SIZE_M>(A, ldA, B, ldB, C, ldC, M, N, K);
}

// Explicit template instantiation
template void GEMM_kernel_v1<__nv_bfloat16, float>(const __nv_bfloat16*, size_t, const __nv_bfloat16*, size_t, float*, size_t, size_t, size_t, size_t);


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

    GEMM_kernel_v1(A, K, B, N, C, N, M, N, K);

    // Cleanup
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}

#endif



#endif