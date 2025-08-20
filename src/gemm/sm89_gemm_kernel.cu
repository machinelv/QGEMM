#include <iostream>
#include <cstdlib>

#include "gpu_lib.h"
#include "gpu_types.h"
#include "fast_quat.h"


namespace gemm_kernel {
namespace normal_gemm {
namespace sm80 {

template<
    size_t BLOCK_M, size_t BLOCK_N, size_t BLOCK_K,
    size_t GROUP_M, size_t STAGE >
struct GEMM_Config {
    size_t BLOCK_TILE_SIZE_M = BLOCK_M;   
    size_t BLOCK_TILE_SIZE_N = BLOCK_N;
    size_t BLOCK_TILE_SIZE_K = BLOCK_K;
    size_t GROUP_SIZE_M      = GROUP_M;
    size_t STAGE_NUMS        = STAGE;
};
struct WMMA_Config {
    static constexpr size_t WARP_TILE_SIZE_M  = 32;    // wmma number of rows in a warp
    static constexpr size_t WARP_TILE_SIZE_N  = 32;    // wmma number of columns in a warp
    static constexpr size_t WMMA_TILE_SIZE_M   = 16;
    static constexpr size_t WMMA_TILE_SIZE_N   = 16;
    static constexpr size_t WMMA_TILE_SIZE_K   = 32;
}


constexpr GEMM_Config<256, 256, 64, 2, 0> gemmconf_256x256x64;
constexpr GEMM_Config<128, 128, 64, 2, 0> gemmconf_128x128x64;
constexpr GEMM_Config<64, 128, 64, 2, 0> gemmconf_64x128x64;
constexpr GEMM_Config<64, 64, 64, 2, 0> gemmconf_64x64x64;




template <typename T, typename T_OUTPUT,
            size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
            size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N, 
            size_t WMMA_TILE_PER_WARP_M, size_t WMMA_TILE_PER_WARP_N, size_t WMMA_TILE_PER_WARP_K,
            size_t THREAD_TILE_SIZE_M_SCALE, size_t THREAD_TILE_SIZE_N_SCALE,
            size_t WMMA_TILE_SIZE_M, size_t WMMA_TILE_SIZE_N, size_t WMMA_TILE_SIZE_K>
inline __device__ void process_data_from_shared_memory_using_wmma(
        const T A_T_shared_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M],
        const T B_shared_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N],
        float32_t A_scale_thread_tile[THREAD_TILE_SIZE_M_SCALE], float32_t B_scale_thread_tile[THREAD_TILE_SIZE_N_SCALE],
        wmma::fragment<matrix_a, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, T, col_major> 
        a_frag[WMMA_TILE_PER_WARP_M],
        wmma::fragment<matrix_b, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, T, row_major>
        b_frag[WMMA_TILE_PER_WARP_N],
        wmma::fragment<accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, float32_t> 
        acc_frag_fp32[WMMA_TILE_PER_WARP_M][WMMA_TILE_PER_WARP_N],
        size_t M, size_t N, size_t K,
        size_t warp_m_id, size_t warp_n_id
    )
{
    wmma::fragment<accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, float32_t> acc_frag_fp32_local[WMMA_TILE_PER_WARP_M][WMMA_TILE_PER_WARP_N];
    #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_M)
    for (size_t wmma_tile_m_idx{0U}; wmma_tile_m_idx < WMMA_TILE_PER_WARP_M; ++wmma_tile_m_idx){
        #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_N)
        for (size_t wmma_tile_n_idx{0U}; wmma_tile_n_idx < WMMA_TILE_PER_WARP_N;++wmma_tile_n_idx) {
            wmma::fill_fragment(acc_frag_fp32_local[wmma_tile_m_idx][wmma_tile_n_idx], static_cast<float32_t>(0));
        }
    }

    // process wmma tile
    #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_K)
    for (size_t wmma_tile_idx_k{0U}; wmma_tile_idx_k < WMMA_TILE_PER_WARP_K; ++wmma_tile_idx_k) {
        // Load data from shared memory to register
        size_t block_tile_wmma_tile_k_idx{wmma_tile_idx_k * WMMA_TILE_SIZE_K};
        // Load A and B matrices from shared memory to registers
        #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_M)
        for (size_t wmma_tile_idx_m{0U}; wmma_tile_idx_m < WMMA_TILE_PER_WARP_M; ++wmma_tile_idx_m) {
            size_t block_tile_wmma_tile_m_idx{warp_m_id * WARP_TILE_SIZE_M + wmma_tile_idx_m * WMMA_TILE_SIZE_M};
            wmma::load_matrix_sync(a_frag[wmma_tile_idx_m], &A_T_shared_block_tile[block_tile_wmma_tile_k_idx][block_tile_wmma_tile_m_idx], BLOCK_TILE_SIZE_M);
        }
        #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_N)
        for (size_t wmma_tile_idx_n{0U}; wmma_tile_idx_n < WMMA_TILE_PER_WARP_N; ++wmma_tile_idx_n) {
            size_t block_tile_wmma_tile_n_idx{warp_n_id * WARP_TILE_SIZE_N + wmma_tile_idx_n * WMMA_TILE_SIZE_N};
            wmma::load_matrix_sync(b_frag[wmma_tile_idx_n], &B_shared_block_tile[block_tile_wmma_tile_k_idx][block_tile_wmma_tile_n_idx], BLOCK_TILE_SIZE_N);
        }

        // Compute the acc_frag
        #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_M)
        for (size_t wmma_tile_idx_m{0U}; wmma_tile_idx_m < WMMA_TILE_PER_WARP_M; ++wmma_tile_idx_m) {
            #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_N)
            for (size_t wmma_tile_idx_n{0U}; wmma_tile_idx_n < WMMA_TILE_PER_WARP_N; ++wmma_tile_idx_n) {
                wmma::mma_sync(acc_frag_fp32_local[wmma_tile_idx_m][wmma_tile_idx_n], a_frag[wmma_tile_idx_m], b_frag[wmma_tile_idx_n], acc_frag_fp32_local[wmma_tile_idx_m][wmma_tile_idx_n]);
            }
        }
    }
    __syncthreads();

    // Scale the acc_frag_fp32 using A_scale and B_scale
    #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_M)
    for (size_t wmma_tile_idx_m{0U}; wmma_tile_idx_m < WMMA_TILE_PER_WARP_M; ++wmma_tile_idx_m) {
        size_t acc_group_start = wmma_tile_idx_m * FP8_ACC_REG_REG_NUMBER;
        #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_N)
        for (size_t wmma_tile_idx_n{0U}; wmma_tile_idx_n < WMMA_TILE_PER_WARP_N; ++wmma_tile_idx_n) {
            // acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n] *= A_scale_thread_tile[0] * B_scale_thread_tile[0];
            // 16 x 16 = 256 acc registers
            // 4 registers per thread
            #pragma clang loop unroll_count(FP8_ACC_REG_REG_NUMBER)
            for (size_t reg_idx{0U}; reg_idx < FP8_ACC_REG_REG_NUMBER; ++reg_idx) {
                acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n].x[reg_idx] += acc_frag_fp32_local[wmma_tile_idx_m][wmma_tile_idx_n].x[reg_idx] * A_scale_thread_tile[acc_group_start + reg_idx] * B_scale_thread_tile[0];
            }
        }
    }
}



template <typename T_INPUT, typename T_OUTPUT, 
            size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
            size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N, 
            size_t WMMA_TILE_SIZE_M, size_t WMMA_TILE_SIZE_N, size_t WMMA_TILE_SIZE_K,
            size_t GROUP_SIZE_M>
__global__ void FP8_GEMM_kernel(const T_INPUT* A, const T_INPUT* B, 
                    const float* A_scale, const float* B_scale, 
                    T_OUTPUT* C, 
                    const size_t M, const size_t N, const size_t K
                )
{
    static_assert(BLOCK_TILE_SIZE_K <= gemmconf::SCALE_BLOCK_SIZE, 
                  "BLOCK_TILE_SIZE_K must be less than or equal to SCALE_BLOCK_SIZE");
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
    const size_t block_id = blockIdx.x + blockIdx.y * gridDim.x;
    const size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const size_t warp_id = thread_id / WARP_SIZE;
    const size_t warp_m_id = warp_id / WARP_NUM_N; // warp_row_idx
    const size_t warp_n_id = warp_id % WARP_NUM_N; // warp_col_idx

    size_t block_tile_num_m = (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M;
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


    __shared__ T_INPUT A_T_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M];
    __shared__ T_INPUT B_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N];

    constexpr size_t THREAD_TILE_SIZE_K_SCALE = 1; // ((BLOCK_TILE_SIZE_K + gemmconf::SCALE_BLOCK_SIZE - 1) / gemmconf::SCALE_BLOCK_SIZE);
    constexpr size_t THREAD_TILE_SIZE_N_SCALE = 1; // ((BLOCK_TILE_SIZE_N + gemmconf::SCALE_BLOCK_SIZE - 1) / gemmconf::SCALE_BLOCK_SIZE);
    constexpr size_t THREAD_TILE_SIZE_M_SCALE = FP8_ACC_REG_REG_NUMBER * WMMA_TILE_PER_WARP_M; // (BLOCK_TILE_SIZE_M + THREAD_NUM - 1) / THREAD_NUM;
    float A_scale_thread_tile[THREAD_TILE_SIZE_M_SCALE];          // store in shared_memory, Note: Assuming that BLOCK_TILE_SIZE_K <= SCALE_BLOCK_SIZE
    float B_scale_thread_tile[THREAD_TILE_SIZE_N_SCALE];          // store in register
    
    wmma::fragment<accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, float32_t> acc_frag_fp32[WMMA_TILE_PER_WARP_M][WMMA_TILE_PER_WARP_N];
    #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_M)
    for (size_t wmma_tile_m_idx{0U}; wmma_tile_m_idx < WMMA_TILE_PER_WARP_M; ++wmma_tile_m_idx){
        #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_N)
        for (size_t wmma_tile_n_idx{0U}; wmma_tile_n_idx < WMMA_TILE_PER_WARP_N;++wmma_tile_n_idx) {
            wmma::fill_fragment(acc_frag_fp32[wmma_tile_m_idx][wmma_tile_n_idx], static_cast<float32_t>(0));
        }
    }
    
    for (size_t block_id_k{0}; block_id_k < K; block_id_k += BLOCK_TILE_SIZE_K)
    {  
        wmma::fragment<matrix_a, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, T_INPUT, col_major> a_frag[WMMA_TILE_PER_WARP_M];
        wmma::fragment<matrix_b, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, T_INPUT, row_major> b_frag[WMMA_TILE_PER_WARP_N];
        size_t block_tile_start_k = block_id_k;
        // load A and B matrices from global memory to shared memory
        load_data_from_global_memory_to_shared_memory_transposed_vectorized
            <T_INPUT, BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K,
            WARP_TILE_SIZE_M, WARP_TILE_SIZE_N, THREAD_NUM>
            (A, B, A_T_block_tile, B_block_tile, M, N, K,
             block_tile_start_m, block_tile_start_n, block_tile_start_k, thread_id);
        // load A_scale and B_scale from global memory to registers
        load_data_from_global_memory_to_register<T_INPUT, WARP_TILE_SIZE_M, WARP_TILE_SIZE_N, WMMA_TILE_SIZE_M, THREAD_TILE_SIZE_M_SCALE, THREAD_TILE_SIZE_N_SCALE, THREAD_NUM>
            (A_scale, B_scale, A_scale_thread_tile, B_scale_thread_tile,
             M, N, K, block_tile_start_m, block_tile_start_n, block_tile_start_k, warp_m_id, warp_n_id, thread_id);
        __syncthreads();
        
        // load a_frag and b_frag from shared memory to registers and compute
        process_data_from_shared_memory_using_wmma<T_INPUT, T_OUTPUT, 
            BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K,
            WARP_TILE_SIZE_M, WARP_TILE_SIZE_N, 
            WMMA_TILE_PER_WARP_M, WMMA_TILE_PER_WARP_N, WMMA_TILE_PER_WARP_K,
            THREAD_TILE_SIZE_M_SCALE, THREAD_TILE_SIZE_N_SCALE,
            WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K>
            (A_T_block_tile, B_block_tile, A_scale_thread_tile, B_scale_thread_tile,
                a_frag, b_frag, acc_frag_fp32, M, N, K, warp_m_id, warp_n_id);
        __syncthreads();
    }

    static_assert(THREAD_TILE_SIZE_M_SCALE == 8, "THREAD_TILE_SIZE_M_SCALE must be 8");
    static_assert(THREAD_TILE_SIZE_N_SCALE == 1, "THREAD_TILE_SIZE_N_SCALE must be 1");
    

    // Store the result to global memory
    #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_M)
    for (size_t wmma_tile_idx_m{0U}; wmma_tile_idx_m < WMMA_TILE_PER_WARP_M; ++wmma_tile_idx_m) {
        #pragma clang loop unroll_count(WMMA_TILE_PER_WARP_N)
        for (size_t wmma_tile_idx_n{0U}; wmma_tile_idx_n < WMMA_TILE_PER_WARP_N; ++wmma_tile_idx_n) {
            wmma::fragment<accumulator, WMMA_TILE_SIZE_M, WMMA_TILE_SIZE_N, WMMA_TILE_SIZE_K, T_OUTPUT> acc_frag;
            #pragma clang loop unroll_count(FP8_ACC_REG_REG_NUMBER)
            for (size_t reg_idx{0U}; reg_idx < FP8_ACC_REG_REG_NUMBER; ++reg_idx) {
                acc_frag.x[reg_idx] = static_cast<T_OUTPUT>(acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n].x[reg_idx]);
                // acc_frag[wmma_tile_idx_m][wmma_tile_idx_n].x[reg_idx] += static_cast<T_OUTPUT>(acc_frag_fp32[wmma_tile_idx_m][wmma_tile_idx_n].x[reg_idx]);
            }
            __syncthreads();
            size_t M_idx = block_tile_start_m + warp_m_id * WARP_TILE_SIZE_M + wmma_tile_idx_m * WMMA_TILE_SIZE_M;
            size_t N_idx = block_tile_start_n + warp_n_id * WARP_TILE_SIZE_N + wmma_tile_idx_n * WMMA_TILE_SIZE_N;
            if (M_idx < M && N_idx < N) {
                wmma::store_matrix_sync(
                    &C[M_idx * N + N_idx], acc_frag, N, wmma::mem_row_major);
            }
        }
    }
}


template<size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K, size_t GROUP_SIZE_M>
void launch_fp8_kernel(const float8_fnuz_t* A, const float8_fnuz_t* B, const float* as, const float* bs,
                       bfloat16_t* C, size_t M, size_t N, size_t K) 
{

    constexpr size_t WARP_NUM_M{BLOCK_TILE_SIZE_M / gemmconf::WARP_TILE_SIZE_M};
    constexpr size_t WARP_NUM_N{BLOCK_TILE_SIZE_N / gemmconf::WARP_TILE_SIZE_N};
    constexpr size_t WARP_NUM{WARP_NUM_M * WARP_NUM_N};
    constexpr size_t THREAD_NUM{WARP_NUM * WARP_SIZE};

#ifdef DEBUG
    std::cout << "Launching kernel with BLOCK_TILE_SIZE_M: " << BLOCK_TILE_SIZE_M
         << ", BLOCK_TILE_SIZE_N: " << BLOCK_TILE_SIZE_N
         << ", BLOCK_TILE_SIZE_K: " << BLOCK_TILE_SIZE_K
         << ", GROUP_SIZE_M: " << GROUP_SIZE_M
         << ", WARP_NUM_M: " << WARP_NUM_M
         << ", WARP_NUM_N: " << WARP_NUM_N
         << ", THREAD_NUM: " << THREAD_NUM
         << std::endl;

    const float8_fnuz_t test[16] = {1,1,1,1,1,1,1,1, 
                              1,1,1,1,1,1,1,1};

    int4 test_int4 = *reinterpret_cast<int4 const*>(&test[0]);
    union {
        float8_fnuz_t fp8_vals[16];
        int4 int4_vals;
    } test_union;
    test_union.int4_vals = test_int4;
    for (size_t i = 0; i < 2; ++i) {
        std::cout << "test_union.fp8_vals[" << i << "]: " << static_cast<float8_fnuz_t>(test_union.fp8_vals[i]) << std::endl;
    }
#endif // DEBUG
    // static_assert(THREAD_NUM <= 1024, "THREAD_NUM must be less than or equal to 1024");

    FP8_GEMM_kernel<float8_fnuz_t, bfloat16_t, 
                    BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K, 
                    gemmconf::WARP_TILE_SIZE_M, gemmconf::WARP_TILE_SIZE_N, 
                    gemmconf::WMMA_TILE_SIZE_M, gemmconf::WMMA_TILE_SIZE_N, gemmconf::WMMA_TILE_SIZE_K, 
                    GROUP_SIZE_M>
        <<<dim3((N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N, (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M, 1),
           dim3(THREAD_NUM), 0>>>(A, B, as, bs, C, M, N, K);
}

/*
Reference implementation of block-scale fp8 gemm
Args:
    data: Tuple that expands to:
        a: torch.Tensor[float8_e4m3fnuz] of shape [m, k], col-major
        b: torch.Tensor[float8_e4m3fnuz] of shape [n, k], col-major 
                        -> same as [k,n] row-major
        a_scale: torch.Tensor[float32] of shape [m, k // SCALE_BLOCK_SIZE], col-major
        b_scale: torch.Tensor[float32] of shape [n // SCALE_BLOCK_SIZE, k // SCALE_BLOCK_SIZE], col-major 
                        -> same as [k // SCALE_BLOCK_SIZE, n // SCALE_BLOCK_SIZE] row-major
        c: torch.Tensor[bfloat16] of shape [m, n], row-major
        SCALE_BLOCK_SIZE = 128
Returns:
    Tensor containing output in bf16
*/
template<size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t GROUP_SIZE_M>
void select_BLOCK_TILE_SIZE_K(const float8_fnuz_t* A, const float8_fnuz_t* B, const float* as, const float* bs,
                       bfloat16_t* C, size_t M, size_t N, size_t K)
{
    constexpr std::array<size_t, 3> BLOCK_TILE_SIZE_K_LIST = {128, 64, 32};

    // if (M / K > 6 && N / K > 6) {
    //     launch_fp8_kernel<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, 64, GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    // } else if (BLOCK_TILE_SIZE_M <= 64 && BLOCK_TILE_SIZE_N <= 64) {
    //     launch_fp8_kernel<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, 128, GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    // } else 
    if (K % BLOCK_TILE_SIZE_K_LIST[0] == 0) { // 128
        launch_fp8_kernel<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K_LIST[0], GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    } else if (K % BLOCK_TILE_SIZE_K_LIST[1] == 0) { // 64
        launch_fp8_kernel<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K_LIST[1], GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    } else if (K % BLOCK_TILE_SIZE_K_LIST[2] == 0) { // 32
        launch_fp8_kernel<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, BLOCK_TILE_SIZE_K_LIST[2], GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    } else {
        throw std::runtime_error("K (" + std::to_string(K) + ") is not divisible by any predefined BLOCK_TILE_SIZE_K in the list.");   
    }
}


template<size_t BLOCK_TILE_SIZE_M, size_t GROUP_SIZE_M>
void select_BLOCK_TILE_SIZE_N(const float8_fnuz_t* A, const float8_fnuz_t* B, const float* as, const float* bs,
                       bfloat16_t* C, size_t M, size_t N, size_t K)
{
    constexpr std::array<size_t, 4> BLOCK_TILE_SIZE_N_LIST = {64, 32};

    // if (M / N > 6) {
    //     select_BLOCK_TILE_SIZE_K<256, 64, GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    // } else if (N / M > 6) {
    //     select_BLOCK_TILE_SIZE_K<64, 256, GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    // } else 
    if (N % BLOCK_TILE_SIZE_N_LIST[0] == 0) { // 256
        select_BLOCK_TILE_SIZE_K<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N_LIST[0], GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    } else if (N % BLOCK_TILE_SIZE_N_LIST[1] == 0) { // 128
        select_BLOCK_TILE_SIZE_K<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N_LIST[1], GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    // } else if (N % BLOCK_TILE_SIZE_N_LIST[2] == 0) { // 64
    //     select_BLOCK_TILE_SIZE_K<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N_LIST[2], GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    // } else if (N % BLOCK_TILE_SIZE_N_LIST[3] == 0) { // 32
    //     select_BLOCK_TILE_SIZE_K<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N_LIST[3], GROUP_SIZE_M>(A, B, as, bs, C, M, N, K);
    } else {
        throw std::runtime_error("N (" + std::to_string(N) + ") is not divisible by any predefined BLOCK_TILE_SIZE_N in the list.");   
    }
}

void select_BLOCK_TILE_SIZE_M(const float8_fnuz_t* A, const float8_fnuz_t* B, const float* as, const float* bs,
                       bfloat16_t* C, size_t M, size_t N, size_t K)
{
    constexpr std::array<size_t, 4> BLOCK_TILE_SIZE_M_LIST = {128, 64, 32};
    
    if (M % BLOCK_TILE_SIZE_M_LIST[0] == 0) { // 128
        select_BLOCK_TILE_SIZE_N<BLOCK_TILE_SIZE_M_LIST[0], 1>(A, B, as, bs, C, M, N, K);
    } else if (M % BLOCK_TILE_SIZE_M_LIST[1] == 0) { // 64
        select_BLOCK_TILE_SIZE_N<BLOCK_TILE_SIZE_M_LIST[1], 1>(A, B, as, bs, C, M, N, K);
    } else if (M % BLOCK_TILE_SIZE_M_LIST[2] == 0) { // 32
        select_BLOCK_TILE_SIZE_N<BLOCK_TILE_SIZE_M_LIST[2], 1>(A, B, as, bs, C, M, N, K);
    // } else if (M % BLOCK_TILE_SIZE_M_LIST[3] == 0) { // 32
    //     select_BLOCK_TILE_SIZE_N<BLOCK_TILE_SIZE_M_LIST[3], 1>(A, B, as, bs, C, M, N, K);
    } else {
        throw std::runtime_error("M (" + std::to_string(M) + ") is not divisible by any predefined BLOCK_TILE_SIZE_M in the list.");   
    }
}


void bf16_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) 
{
    const size_t m = a.size(0);
    const size_t n = b.size(0);
    const size_t k = a.size(1); 

    select_BLOCK_TILE_SIZE_M(static_cast<bfloat16_t*>(a.data_ptr()), static_cast<bfloat16_t*>(b.data_ptr()), 
    as.data_ptr<float>(), bs.data_ptr<float>(), static_cast<bfloat16_t*>(c.data_ptr()), m, n, k);
}

void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) 
{
    const size_t m = a.size(0);
    const size_t n = b.size(0);
    const size_t k = a.size(1); 

    select_BLOCK_TILE_SIZE_M(static_cast<float8_fnuz_t*>(a.data_ptr()), static_cast<float8_fnuz_t*>(b.data_ptr()), 
    as.data_ptr<float>(), bs.data_ptr<float>(), static_cast<bfloat16_t*>(c.data_ptr()), m, n, k);
}

}
}
}
