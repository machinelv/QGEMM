#pragma once


template <int BLOCK_SZ, int NUM_PER_THREAD>
__global__ void mat_transpose_kernel(const float* idata, float* odata, int M, int N) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ+1];

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;

    constexpr int ROW_STRIDE = BLOCK_SZ / NUM_PER_THREAD;

    if (x < N) {
        if (y + BLOCK_SZ <= M) {
            #pragma unroll
            for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
                sdata[ty + y_off][tx] = idata[(y + y_off) * N + x]; 
            }
        } else {
            for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
                if (ty + y_off < M) {
                    sdata[ty + y_off][tx] = idata[(y + y_off) * N + x];
                }
            }
        }

    }
    __syncthreads();

    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;
    if (x < M) {
        if (y + BLOCK_SZ <= N) {
            #pragma unroll
            for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
                odata[(y + y_off) * M + x] = sdata[tx][ty + y_off];
            }
        } else {
            for (int y_off = 0; y_off < BLOCK_SZ; y_off += ROW_STRIDE) {
                if (y + y_off < N) {
                    odata[(y + y_off) * M + x] = sdata[tx][ty + y_off];
                }
            }
        }
    }
}

void mat_transpose_v4(const float* idata, float* odata, int M, int N) {
    constexpr int BLOCK_SZ = 32;
    constexpr int NUM_PER_THREAD = 4;
    dim3 block(BLOCK_SZ, BLOCK_SZ/NUM_PER_THREAD);
    dim3 grid(Ceil(N, BLOCK_SZ), Ceil(M, BLOCK_SZ));
    mat_transpose_kernel_v4<BLOCK_SZ, NUM_PER_THREAD><<<grid, block>>>(idata, odata, M, N);
}
