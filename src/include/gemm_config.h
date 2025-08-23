#pragma once

#include <cstdlib>


namespace gemm_kernel {

struct GEMM_Config {
    size_t BLOCK_TILE_SIZE_M;   
    size_t BLOCK_TILE_SIZE_N;
    size_t BLOCK_TILE_SIZE_K;
    size_t STAGE_NUMS;
    size_t GROUP_SIZE_M;
};
namespace sm80{
struct WMMA_BF16_Config {
    static constexpr size_t WARP_TILE_SIZE_M  = 64;    // wmma number of rows in a warp  
    static constexpr size_t WARP_TILE_SIZE_N  = 64;    // wmma number of columns in a warp
    static constexpr size_t WMMA_TILE_SIZE_M   = 16;
    static constexpr size_t WMMA_TILE_SIZE_N   = 16;
    static constexpr size_t WMMA_TILE_SIZE_K   = 16;
};
constexpr GEMM_Config config_128_128_64_1_1{128, 128, 64, 1, 1};

} // sm80

namespace sm89{
struct WMMA_BF16_Config {
    static constexpr size_t WARP_TILE_SIZE_M  = 64;    // wmma number of rows in a warp  
    static constexpr size_t WARP_TILE_SIZE_N  = 64;    // wmma number of columns in a warp
    static constexpr size_t WMMA_TILE_SIZE_M   = 16;
    static constexpr size_t WMMA_TILE_SIZE_N   = 16;
    static constexpr size_t WMMA_TILE_SIZE_K   = 16;
};

struct WMMA_FP8_Config {
    static constexpr size_t WARP_TILE_SIZE_M  = 64;    // wmma number of rows in a warp  
    static constexpr size_t WARP_TILE_SIZE_N  = 64;    // wmma number of columns in a warp
    static constexpr size_t WMMA_TILE_SIZE_M   = 16;
    static constexpr size_t WMMA_TILE_SIZE_N   = 16;
    static constexpr size_t WMMA_TILE_SIZE_K   = 16;
};

constexpr GEMM_Config config_64_64_64_1_4{64, 64, 64, 1, 4};
constexpr GEMM_Config config_128_64_64_1_4{128, 64, 64, 1, 4};
constexpr GEMM_Config config_128_128_64_1_1{128, 128, 64, 1, 1};
constexpr GEMM_Config config_128_128_64_1_4{128, 128, 64, 1, 4};
constexpr GEMM_Config config_128_128_64_1_8{128, 128, 64, 1, 8};
constexpr GEMM_Config config_256_128_128_1_4{256, 128, 64, 1, 4};


} // sm89

} // gemm_kernel