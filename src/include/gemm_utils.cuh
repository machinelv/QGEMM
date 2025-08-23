#pragma once
#include "common.h"
#include "gpu_type.h"

#define LDMATRIX_BF16_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.bf16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))
#define LDMATRIX_BF16_TRANSPOSE_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.bf16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define BF16MMA16816(Rd0, Rd1, Ra0, Ra1, Ra2, Ra3, Rb0, Rb1, Rc0, Rc1)         \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.bf16.bf16.bf16.bf16         \
                    {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"      \
                 : "=r"(Rd0), "=r"(Rd1)                                     \
                 : "r"(Ra0), "r"(Ra1), "r"(Ra2), "r"(Ra3),                  \
                   "r"(Rb0), "r"(Rb1),                                      \
                   "r"(Rc0), "r"(Rc1))

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

namespace gemm_kernel {

/**
 * Swizzle the coordinates for shared memory access
 * @template parameters
 * B: 2^B rows in total
 * M: 2^M * 16 bits per thread_block
 * S: 2^S
 */
template <size_t B, size_t S, size_t M, typename ElemType>
inline __device__ void shared_memory_swizzle_coordinate(const size_t lane_id, size_t &swizzled_x, size_t &swizzled_y)
{
    // constexpr size_t element_per_threadblock = (1 << M) * 2 / sizeof(ElemType);
    // constexpr size_t num_rows = 1 << B;
    // constexpr size_t 

    // col:
    swizzled_y = 


}

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

#if CUDA_ARCH >= 80

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


#endif


}