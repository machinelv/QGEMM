#include <iostream>
#include <cstdlib>

#include "gpu_lib.h"
#include "gpu_type.h"
#include "matrix_transpose.h"


#define VECTYPE float4

template<size_t BLOCK_TILE_SIZE_M, size_t BLOCK_TILE_SIZE_N, size_t THREAD_NUM>
__global__ void matrix_transpose_kernel_v3(const __nv_bfloat16* idata, __nv_bfloat16* odata, int M, int N)
{
    // Get index
    size_t block_id_m = blockIdx.y;
    size_t block_id_n = blockIdx.x;
    size_t thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // load matrix from global memory to shared memory using vector fetching
    __shared__ __nv_bfloat16 block_tile[BLOCK_TILE_SIZE_N][BLOCK_TILE_SIZE_M + 16];

    static_assert(sizeof(VECTYPE) % sizeof(__nv_bfloat16) == 0, "VECTYPE must be a multiple of T size");
    constexpr size_t NUM_VECTOR_UNITS{sizeof(VECTYPE) / sizeof(__nv_bfloat16)};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_N{BLOCK_TILE_SIZE_N / NUM_VECTOR_UNITS};
    constexpr size_t VECTORIZED_BLOCK_TILE_SIZE_M{BLOCK_TILE_SIZE_M / NUM_VECTOR_UNITS};

    static_assert(BLOCK_TILE_SIZE_M % NUM_VECTOR_UNITS == 0);
    static_assert(BLOCK_TILE_SIZE_N % NUM_VECTOR_UNITS == 0);

    size_t block_tile_start_m = block_id_m * BLOCK_TILE_SIZE_M;
    size_t block_tile_start_n = block_id_n * BLOCK_TILE_SIZE_N;


    size_t block_tile_M_id{(thread_id ) / VECTORIZED_BLOCK_TILE_SIZE_N};
    size_t block_tile_N_id{(thread_id ) % VECTORIZED_BLOCK_TILE_SIZE_N * NUM_VECTOR_UNITS};

    size_t M_id{block_tile_M_id + block_tile_start_m};
    size_t N_id{block_tile_N_id + block_tile_start_n};

    VECTYPE row_vector_vals;

    row_vector_vals = *reinterpret_cast<VECTYPE const*>(&idata[M_id * N + N_id]);
    // *reinterpret_cast<VECTYPE*>(&block_tile[block_tile_M_id][block_tile_N_id]) = row_vector_vals.vec;
    #pragma unroll
    for (size_t i = 0; i < NUM_VECTOR_UNITS; i++) {
        block_tile[block_tile_N_id + i][block_tile_M_id] = reinterpret_cast<__nv_bfloat16 const*>(&row_vector_vals)[i];
    }
    __syncthreads();

    block_tile_N_id = (thread_id) / VECTORIZED_BLOCK_TILE_SIZE_M;
    block_tile_M_id = (thread_id) % VECTORIZED_BLOCK_TILE_SIZE_M * NUM_VECTOR_UNITS;

    M_id = block_tile_M_id + block_tile_start_m;
    N_id = block_tile_N_id + block_tile_start_n;

    row_vector_vals = *reinterpret_cast<VECTYPE const*>(&block_tile[block_tile_N_id][block_tile_M_id]);
    *reinterpret_cast<VECTYPE*>(&odata[N_id * M + M_id]) = row_vector_vals;
}

template <typename T>
void matrix_transpose(const T* idata, T* odata, int M, int N) {
    if constexpr (sizeof(T) == 2) {
        constexpr size_t THREAD_NUM = 256;
        constexpr size_t BLOCK_TILE_SIZE_M = 64;
        constexpr size_t BLOCK_TILE_SIZE_N = 32;
        
        dim3 blockDim(THREAD_NUM, 1, 1);
        dim3 gridDim((N + BLOCK_TILE_SIZE_N - 1) / BLOCK_TILE_SIZE_N, (M + BLOCK_TILE_SIZE_M - 1) / BLOCK_TILE_SIZE_M, 1);
        matrix_transpose_kernel_v3<BLOCK_TILE_SIZE_M, BLOCK_TILE_SIZE_N, THREAD_NUM><<<gridDim, blockDim>>>((__nv_bfloat16*)idata, (__nv_bfloat16*)odata, M, N);
    } else {
        std::cerr << "Only support bf16 data type now." << std::endl;
        return;
    }
}

// Explicit template instantiation for commonly used types
template void matrix_transpose<__nv_bfloat16>(const __nv_bfloat16* idata, __nv_bfloat16* odata, int M, int N);