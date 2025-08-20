#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
#include <iomanip>
#include <random>
#include <memory>
#include <string>
#include <cstdint>
#include <cmath>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_bf16.h>



// Custom GEMM headers
#include "gemm.h"
// Test configuration and utility headers
#include "simple_timer.h"
#include "test_config.h"

int main() {
    std::cout << "CUDA SM80 GEMM Benchmark using CUTLASS" << std::endl;
    std::cout << "Testing precisions: BF16, INT8" << std::endl;
    
    // Check GPU capability
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    if (prop.major < 8) {
        std::cerr << "This benchmark requires SM80+ GPU (Ampere architecture)" << std::endl;
        return -1;
    }
    
    // Test configurations for different sizes
    std::vector<TestConfig> configs_bf16 = {
        TestConfig(512, 512, 512, "bf16"),
        TestConfig(1024, 1024, 1024, "bf16"),
        TestConfig(2048, 2048, 2048, "bf16"),
        TestConfig(4096, 4096, 4096, "bf16"),
    };

    std::vector<TestConfig> configs_int8 = {
        TestConfig(512, 512, 512, "int8"),
        TestConfig(1024, 1024, 1024, "int8"),
        TestConfig(2048, 2048, 2048, "int8"),
        TestConfig(4096, 4096, 4096, "int8"),
    };


    std::vector<CustomGEMMFunction<__nv_bfloat16, __nv_bfloat16>> bf16_functions = {
            CustomGEMMFunction<__nv_bfloat16, __nv_bfloat16>("cutlass_gemm_test", cutlass_gemm_test<__nv_bfloat16, __nv_bfloat16>),
            CustomGEMMFunction<__nv_bfloat16, __nv_bfloat16>("GEMM_kernel_v1", GEMM_kernel_v1<__nv_bfloat16, __nv_bfloat16>)
        };

    std::vector<CustomGEMMFunction<int8_t, int8_t>> int8_functions = {
            CustomGEMMFunction<int8_t, int8_t>("cutlass_gemm_test", cutlass_gemm_test<int8_t, int8_t>),
        };

    GEMMBenchmark<__nv_bfloat16, __nv_bfloat16> bf16_benchmark(configs_bf16, bf16_functions);
    GEMMBenchmark<int8_t, int8_t> int8_benchmark(configs_int8, int8_functions);

    bf16_benchmark.set_reference_function(CustomGEMMFunction<__nv_bfloat16, __nv_bfloat16>("cutlass_gemm_test", cutlass_gemm_test<__nv_bfloat16, __nv_bfloat16>));
    int8_benchmark.set_reference_function(CustomGEMMFunction<int8_t, int8_t>("cutlass_gemm_test", cutlass_gemm_test<int8_t, int8_t>));
    

    bf16_benchmark.run_benchmark();
    int8_benchmark.run_benchmark();

    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "Benchmark completed!" << std::endl;
    std::cout << "Note: Add your custom GEMM implementations to the benchmark by calling add_custom_function()" << std::endl;
    
    return 0;
}