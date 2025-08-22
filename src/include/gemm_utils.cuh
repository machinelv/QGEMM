#pragma once
#include "common.h"
#include "gpu_lib.h"

namespace gemm_kernel {
template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
          size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N,
          size_t THREAD_NUM, typename VECTOR_TYPE = float4>
inline __device__ void load_data_from_global_memory_to_shared_memory(
        const T* A, const T* B, 
        T* A_block_tile, T* B_block_tile,
        size_t M, size_t N, size_t K,
        size_t block_tile_start_m, size_t block_tile_start_n, size_t block_tile_start_k,
        size_t thread_id
    )
{
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0, "VECTOR_TYPE must be a multiple of T size");
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_M{BLOCK_TILE_SIZE_M / NUM_VECTOR_UNITS};

    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_M % NUM_VECTOR_UNITS == 0);
    union VectorAccess {
        VECTOR_TYPE vec;
        T elements[NUM_VECTOR_UNITS];
    };
    
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_M + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        size_t A_block_tile_M_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t A_block_tile_K_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};

        size_t A_M_id{A_block_tile_M_id + block_tile_start_m};
        size_t A_K_id{A_block_tile_K_id + block_tile_start_k};

        VectorAccess A_row_vector_vals;

        A_row_vector_vals.vec = *reinterpret_cast<VECTOR_TYPE const*>(&A[A_M_id * K + A_K_id]);
        if (A_block_tile_K_id < BLOCK_TILE_SIZE_K && A_block_tile_M_id < BLOCK_TILE_SIZE_M) {
            *reinterpret_cast<VECTOR_TYPE*>(&A_block_tile[A_block_tile_K_id + A_block_tile_M_id * BLOCK_TILE_SIZE_K]) = A_row_vector_vals.vec;
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
            *reinterpret_cast<VECTOR_TYPE*>(&B_block_tile[B_block_tile_K_id * BLOCK_TILE_SIZE_N + B_block_tile_N_id]) = B_row_vector_vals.vec;
        }
    }
}



template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
          size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N,
          size_t THREAD_NUM, typename VECTOR_TYPE = float4>
inline __device__ void load_data_from_global_memory_to_shared_memory_transposed(
        const T* A, const T* B, 
        T* A_T_block_tile, T* B_block_tile,
        size_t M, size_t N, size_t K,
        size_t block_tile_start_m, size_t block_tile_start_n, size_t block_tile_start_k,
        size_t thread_id
    )
{
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0, "VECTOR_TYPE must be a multiple of T size");

    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_M{BLOCK_TILE_SIZE_M / NUM_VECTOR_UNITS};

    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_M % NUM_VECTOR_UNITS == 0);
    union VectorAccess {
        VECTOR_TYPE vec;
        T elements[NUM_VECTOR_UNITS];
    };

    
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_M + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        // size_t A_block_tile_K_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_M};
        // size_t A_block_tile_M_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_M * NUM_VECTOR_UNITS};
        size_t A_block_tile_M_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t A_block_tile_K_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};

        size_t A_M_id{A_block_tile_M_id + block_tile_start_m};
        size_t A_K_id{A_block_tile_K_id + block_tile_start_k};

        VectorAccess A_row_vector_vals;

        A_row_vector_vals.vec = *reinterpret_cast<VECTOR_TYPE const*>(&A[A_M_id * K + A_K_id]);
        for (size_t i = 0; i < NUM_VECTOR_UNITS; ++i) {
            if (A_block_tile_K_id < BLOCK_TILE_SIZE_K && A_block_tile_M_id < BLOCK_TILE_SIZE_M) {
                A_T_block_tile[(A_block_tile_K_id + i) * BLOCK_TILE_SIZE_M + A_block_tile_M_id] = A_row_vector_vals.elements[i];
            }
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
            *reinterpret_cast<VECTOR_TYPE*>(&B_block_tile[B_block_tile_K_id * BLOCK_TILE_SIZE_N + B_block_tile_N_id]) = B_row_vector_vals.vec;
        }
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
          size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N,
          size_t THREAD_NUM, 
          typename VECTOR_TYPE = float4>
inline __device__ void load_data_from_global_memory_to_shared_memory_transposed_swizzle(
        const T* A, const T* B, 
        T* A_T_block_tile, T* B_block_tile,
        size_t M, size_t N, size_t K,
        size_t block_tile_start_m, size_t block_tile_start_n, size_t block_tile_start_k,
        size_t thread_id
    )
{
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0, "VECTOR_TYPE must be a multiple of T size");

    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTOR_TYPE) / sizeof(T)};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_M{BLOCK_TILE_SIZE_M / NUM_VECTOR_UNITS};

    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_M % NUM_VECTOR_UNITS == 0);
    union VectorAccess {
        VECTOR_TYPE vec;
        T elements[NUM_VECTOR_UNITS];
    };

    
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_M + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        // size_t A_block_tile_K_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_M};
        // size_t A_block_tile_M_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_M * NUM_VECTOR_UNITS};
        size_t A_block_tile_M_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t A_block_tile_K_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};

        size_t A_M_id{A_block_tile_M_id + block_tile_start_m};
        size_t A_K_id{A_block_tile_K_id + block_tile_start_k};

        VectorAccess A_row_vector_vals;

        A_row_vector_vals.vec = *reinterpret_cast<VECTOR_TYPE const*>(&A[A_M_id * K + A_K_id]);
        for (size_t i = 0; i < NUM_VECTOR_UNITS; ++i) {
            if (A_block_tile_K_id < BLOCK_TILE_SIZE_K && A_block_tile_M_id < BLOCK_TILE_SIZE_M) {
                A_T_block_tile[(A_block_tile_K_id + i) * BLOCK_TILE_SIZE_M + A_block_tile_M_id] = A_row_vector_vals.elements[i];
            }
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
            *reinterpret_cast<VECTOR_TYPE*>(&B_block_tile[B_block_tile_K_id * BLOCK_TILE_SIZE_N + B_block_tile_N_id]) = B_row_vector_vals.vec;
        }
    }
}

#if CUDA_ARCH >= 80

#define copy_cg_async(smem_ptr, gmem_ptr, BTYES)                \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"  \
                :: "r"(smem_ptr),                               \
                "l"(gmem_ptr),                                  \
                "n"(BTYES));

#define copy_async_commit                           \
    asm volatile("cp.async.commit_group;\n" ::);

#define cp_async_wait(N)                                    \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));

#define  cp_async_wait_all      \
    asm volatile("cp.async.wait_all;\n"  ::);

template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
          size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N,
          size_t THREAD_NUM, typename VECTOR_TYPE = float4>
inline __device__ void load_data_from_global_memory_to_shared_memory_async(
        const T* A, const T* B,
        T* A_block_tile, T* B_block_tile,
        size_t M, size_t N, size_t K,
        size_t block_tile_start_m, size_t block_tile_start_n, size_t block_tile_start_k,
        size_t thread_id
    )
{
    constexpr size_t VECTOR_SIZE_BYTE{sizeof(VECTOR_TYPE)};
    static_assert(VECTOR_SIZE_BYTE % sizeof(T) == 0, "VECTOR_TYPE must be a multiple of T size");

    constexpr size_t NUM_VECTOR_UNITS{VECTOR_SIZE_BYTE / sizeof(T)};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_K{BLOCK_TILE_SIZE_K / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_M{BLOCK_TILE_SIZE_M / NUM_VECTOR_UNITS};

    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_M % NUM_VECTOR_UNITS == 0);
    union VectorAccess {
        VECTOR_TYPE vec;
        T elements[NUM_VECTOR_UNITS];
    };
    
    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_M + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        size_t A_block_tile_M_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_K};
        size_t A_block_tile_K_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_K * NUM_VECTOR_UNITS};

        size_t A_M_id{A_block_tile_M_id + block_tile_start_m};
        size_t A_K_id{A_block_tile_K_id + block_tile_start_k};

        uint32_t smem_addr = __cvta_generic_to_shared(&A_block_tile[A_block_tile_K_id + A_block_tile_M_id * BLOCK_TILE_SIZE_K]);
        auto gmem_addr = (&A[A_M_id * K + A_K_id]);
        copy_cg_async(smem_addr, gmem_addr, VECTOR_SIZE_BYTE);
    }

    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_N + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        size_t B_block_tile_K_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_N};
        size_t B_block_tile_N_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_N * NUM_VECTOR_UNITS};

        size_t B_K_id{B_block_tile_K_id + block_tile_start_k};
        size_t B_N_id{B_block_tile_N_id + block_tile_start_n};

        uint32_t smem_addr = __cvta_generic_to_shared(&B_block_tile[B_block_tile_K_id * BLOCK_TILE_SIZE_N + B_block_tile_N_id]);
        auto gmem_addr = (&B[B_K_id * N + B_N_id]);
        copy_cg_async(smem_addr, gmem_addr, VECTOR_SIZE_BYTE);
    }
}

template <typename T, size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t BLOCK_TILE_SIZE_K,
          size_t WARP_TILE_SIZE_M, size_t WARP_TILE_SIZE_N,
          size_t THREAD_NUM, typename VECTOR_TYPE = float4>
inline __device__ void load_data_from_global_memory_to_shared_memory_swizzle_async(
         const T* A, const T* B, 
        T* A_block_tile, T* B_block_tile,
        size_t M, size_t N, size_t K,
        size_t block_tile_start_m, size_t block_tile_start_n, size_t block_tile_start_k,
        size_t thread_id
    )
{
    constexpr size_t VECTOR_SIZE_BYTE{sizeof(VECTOR_TYPE)};
    static_assert(VECTOR_SIZE_BYTE % sizeof(T) == 0, "VECTOR_TYPE must be a multiple of T size");

    constexpr size_t NUM_VECTOR_UNITS{VECTOR_SIZE_BYTE / sizeof(T)};
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

        // VectorAccess A_row_vector_vals;

        // A_row_vector_vals.vec = *reinterpret_cast<VECTOR_TYPE const*>(&A[A_M_id + A_K_id * M]);
        // if (A_block_tile_K_id < BLOCK_TILE_SIZE_K && A_block_tile_M_id < BLOCK_TILE_SIZE_M) {
        //     *reinterpret_cast<VECTOR_TYPE*>(&A_block_tile[A_block_tile_K_id + A_block_tile_M_id * BLOCK_TILE_SIZE_K]) = A_row_vector_vals.vec;
        // }
        copy_cg_async(&A_block_tile[A_block_tile_K_id + A_block_tile_M_id * BLOCK_TILE_SIZE_K], &A[A_M_id + A_K_id * M], VECTOR_SIZE_BYTE);
    }

    for (size_t load_idx{0U}; load_idx < (BLOCK_TILE_SIZE_K * VECTORIZED_BLOCK_TILE_SIZE_N + THREAD_NUM - 1) / THREAD_NUM; load_idx ++) {
        size_t B_block_tile_K_id{(thread_id + load_idx * THREAD_NUM) / VECTORIZED_BLOCK_TILE_SIZE_N};
        size_t B_block_tile_N_id{(thread_id + load_idx * THREAD_NUM) % VECTORIZED_BLOCK_TILE_SIZE_N * NUM_VECTOR_UNITS};

        size_t B_K_id{B_block_tile_K_id + block_tile_start_k};
        size_t B_N_id{B_block_tile_N_id + block_tile_start_n};

        // VectorAccess B_row_vector_vals;

        // B_row_vector_vals.vec = *reinterpret_cast<VECTOR_TYPE const*>(&B[B_K_id * N + B_N_id]);
        // if (B_block_tile_N_id < BLOCK_TILE_SIZE_N && B_block_tile_K_id < BLOCK_TILE_SIZE_K) {
        //     *reinterpret_cast<VECTOR_TYPE*>(&B_block_tile[B_block_tile_K_id * BLOCK_TILE_SIZE_N + B_block_tile_N_id]) = B_row_vector_vals.vec;
        // }
        copy_cg_async(&B_block_tile[B_block_tile_K_id * BLOCK_TILE_SIZE_N + B_block_tile_N_id], &B[B_K_id * N + B_N_id], VECTOR_SIZE_BYTE);
    }
}

#endif


}