#include "gpu_lib.h"


namespace gemm_kernel{



template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
          size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N,
          size_t THREAD_NUM, 
          typename VECTOR_TYPE = char4>
inline __device__ void load_data_from_global_memory_to_shared_memory_transposed_vectorized(
        const T* A, const T* B, 
        T A_T_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_M], T B_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_N],
        size_t M, size_t N, size_t K,
        size_t block_tile_start_m, size_t block_tile_start_n, size_t block_tile_start_k,
        size_t thread_id
    )
{
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0, "VECTOR_TYPE must be a multiple of T size");

    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_M{BLOCK_TILE_SIZE_M / NUM_VECTOR_UNITS};

    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_M % NUM_VECTOR_UNITS == 0);
    union VectorAccess {
        VECTOR_TYPE vec;
        T elements[NUM_VECTOR_UNITS];
    };
    
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_M + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        size_t A_block_tile_K_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_M};
        size_t A_block_tile_M_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_M * NUM_VECTOR_UNITS};

        size_t A_M_id{A_block_tile_M_id + block_tile_start_m};
        size_t A_K_id{A_block_tile_K_id + block_tile_start_k};

        VectorAccess A_row_vector_vals;

        A_row_vector_vals.vec = *reinterpret_cast<VECTOR_TYPE const*>(&A[A_M_id + A_K_id * M]);
        if (A_block_tile_K_id < BLOCK_TILE_SIZE_K && A_block_tile_M_id < BLOCK_TILE_SIZE_M) {
            *reinterpret_cast<VECTOR_TYPE*>(&A_T_block_tile[A_block_tile_K_id][A_block_tile_M_id]) = A_row_vector_vals.vec;
        }
    }

    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_N + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        size_t B_block_tile_K_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_N};
        size_t B_block_tile_N_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_N * NUM_VECTOR_UNITS};

        size_t B_K_id{B_block_tile_K_id + block_tile_start_k};
        size_t B_N_id{B_block_tile_N_id + block_tile_start_n};

        VectorAccess B_row_vector_vals;

        B_row_vector_vals.vec = *reinterpret_cast<VECTOR_TYPE const*>(&B[B_K_id * N + B_N_id]);
        if (B_block_tile_N_id < BLOCK_TILE_SIZE_N && B_block_tile_K_id < BLOCK_TILE_SIZE_K) {
            *reinterpret_cast<VECTOR_TYPE*>(&B_block_tile[B_block_tile_K_id][B_block_tile_N_id]) = B_row_vector_vals.vec;
        }
    }
    
}


template <typename T, size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N, size_t WMMA_TILE_SIZE_M, size_t THREAD_TILE_SIZE_M_SCALE, size_t THREAD_TILE_SIZE_N_SCALE, size_t THREAD_NUM>
inline __device__ void load_data_from_global_memory_to_register(
        const T* A_scale, const T* B_scale,
        T A_scale_thread_tile[THREAD_TILE_SIZE_M_SCALE], T B_scale_thread_tile[THREAD_TILE_SIZE_N_SCALE],
        size_t M, size_t N, size_t K,
        size_t block_tile_offset_m, size_t block_tile_offset_n, size_t block_tile_offset_k,
        size_t warp_m_id, size_t warp_n_id, size_t thread_id
    )
{
    static_assert(THREAD_TILE_SIZE_M_SCALE == 8 || THREAD_TILE_SIZE_M_SCALE == 4 , "THREAD_TILE_SIZE_M_SCALE must be 8");
    static_assert(THREAD_TILE_SIZE_N_SCALE == 1, "THREAD_TILE_SIZE_N_SCALE must be 1");

    const size_t M_SCALE = M;
    const size_t N_SCALE = (N + gemmconf::SCALE_BLOCK_SIZE - 1) / gemmconf::SCALE_BLOCK_SIZE;

    // size_t warp_id{thread_id / WARP_SIZE};
    size_t lane_id{thread_id % WARP_SIZE};
    size_t acc_group_id{lane_id / FP8_ACC_REG_LANE_GROUP};

    size_t scale_k_id{block_tile_offset_k / gemmconf::SCALE_BLOCK_SIZE};
    size_t scale_n_id{(block_tile_offset_n + warp_n_id * WARP_TILE_SIZE_N) / gemmconf::SCALE_BLOCK_SIZE};
    // TODO: Find how acc fragment stored in a warp
    size_t scale_m_start{block_tile_offset_m + warp_m_id * WARP_TILE_SIZE_M + acc_group_id * FP8_ACC_REG_REG_NUMBER};
    

    A_scale_thread_tile[0] = A_scale[(scale_m_start)     + scale_k_id * M_SCALE];
    A_scale_thread_tile[1] = A_scale[(scale_m_start + 1) + scale_k_id * M_SCALE];
    A_scale_thread_tile[2] = A_scale[(scale_m_start + 2) + scale_k_id * M_SCALE];
    A_scale_thread_tile[3] = A_scale[(scale_m_start + 3) + scale_k_id * M_SCALE];

    if constexpr (THREAD_TILE_SIZE_M_SCALE == 8) {
        scale_m_start += WMMA_TILE_SIZE_M;
        A_scale_thread_tile[4] = A_scale[(scale_m_start)     + scale_k_id * M_SCALE];
        A_scale_thread_tile[5] = A_scale[(scale_m_start + 1) + scale_k_id * M_SCALE];
        A_scale_thread_tile[6] = A_scale[(scale_m_start + 2) + scale_k_id * M_SCALE];
        A_scale_thread_tile[7] = A_scale[(scale_m_start + 3) + scale_k_id * M_SCALE];
    }

    // Load B_scale from global memory to register
    if constexpr (THREAD_TILE_SIZE_N_SCALE == 1) {
        B_scale_thread_tile[0] = B_scale[(scale_n_id + 0) + scale_k_id * N_SCALE];
    } else if constexpr (THREAD_TILE_SIZE_N_SCALE == 2) {
        B_scale_thread_tile[0] = B_scale[(scale_n_id + 0) + scale_k_id * N_SCALE];
        B_scale_thread_tile[1] = B_scale[(scale_n_id + 1) + scale_k_id * N_SCALE];
    } // It is impossible to have THREAD_TILE_SIZE_N_SCALE > 2 or even THREAD_TILE_SIZE_N_SCALE = 2
}



template <typename T, size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N, size_t WMMA_TILE_SIZE_M, size_t THREAD_TILE_SIZE_M_SCALE, size_t THREAD_TILE_SIZE_N_SCALE, size_t THREAD_NUM>
inline __device__ void load_data_from_register_to_shared_memory(
        const float* A_scale, const float* B_scale,
        float A_scale_thread_tile[THREAD_TILE_SIZE_M_SCALE], float B_scale_thread_tile[THREAD_TILE_SIZE_N_SCALE],
        size_t M, size_t N, size_t K,
        size_t block_tile_offset_m, size_t block_tile_offset_n, size_t block_tile_offset_k,
        size_t warp_m_id, size_t warp_n_id, size_t thread_id
    )
{


}

}