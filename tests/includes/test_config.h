#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <random>
#include <iomanip>
#include <cmath>
#include <memory>

#include <cuda_runtime.h>

#include "simple_timer.h"
#include "gemm.h"
#include "blas.h"

template <typename T>
auto to_cuda_dtype() {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return CUDA_R_16BF;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return CUDA_R_8I;
    } else if constexpr (std::is_same_v<T, float>) {
        return CUDA_R_32F;
    } 
    return CUDA_R_32F;
}

template <typename typeIn, typename typeOut>
void reference_gemm_blas(typeIn* A, size_t ldA,
                        typeIn* B, size_t ldB, 
                        typeOut* C, size_t ldC,
                        size_t M, size_t N, size_t K) {
    auto in_dtype  = to_cuda_dtype<typeIn>();
    auto out_dtype = to_cuda_dtype<typeOut>();
    const float alpha = 1.0f;
    const float beta = 0.0f;
                            
    cublasGemmEx(blas::blas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, in_dtype, ldB,
        A, in_dtype, ldA,
        &beta,
        C, out_dtype, ldC,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT);
    // if (status != CUBLAS_STATUS_SUCCESS) {
    //     std::cerr << "cuBLAS GEMM (bf16->bf16) failed: " << status << std::endl;
    // }
}

// Test configuration for SM80 architecture
struct TestConfig {
    size_t M, N, K;
    std::string precision;
    size_t warmup_runs;
    size_t benchmark_runs;
    int seed; // For reproducibility
    double tolerance;

    TestConfig(size_t m, size_t n, size_t k, const std::string& prec,
               size_t warmup = 5, size_t bench = 10, double tol = 5e-2, int s = 42)
        : M(m), N(n), K(k), precision(prec), warmup_runs(warmup),
          benchmark_runs(bench), seed(s), tolerance(tol){}
};

template<typename typeIn, typename typeOut>
class CustomGEMMFunction {
public:
    std::string name;
    std::function<void(typeIn*, size_t, typeIn*, size_t, typeOut*, size_t, size_t, size_t, size_t)> func;
    
    // Default constructor
    CustomGEMMFunction() : name(""), func(nullptr) {}
    
    CustomGEMMFunction(const std::string& n, decltype(func) f) 
        : name(n), func(f) {}
    void operator () (typeIn* A, size_t ldA, typeIn* B, size_t ldB, typeOut* C, size_t ldC, size_t M, size_t N, size_t K) const {
        if (func) {
            func(A, ldA, B, ldB, C, ldC, M, N, K);
        } else {
            throw std::runtime_error("Custom GEMM function is not set.");
        }
    }
};


template<typename typeIn, typename typeOut>
class GEMMBenchmark{
private:
    std::vector<CustomGEMMFunction<typeIn, typeOut>> custom_functions;
    CustomGEMMFunction<typeIn, typeOut> reference_function;
    std::vector<TestConfig> configs;
private:
    void cleanup_memory(typeIn* d_A, typeIn* d_B, typeOut* d_C, typeOut* d_D, typeOut* d_ref,
                    typeIn* h_A, typeIn* h_B, typeOut* h_C, typeOut* h_ref) {
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        delete[] h_ref;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_D);
        cudaFree(d_ref);
    }

    double run_func(const CustomGEMMFunction<typeIn, typeOut>& func, 
                    typeIn* d_A, size_t ldA,
                    typeIn* d_B, size_t ldB,
                    typeOut* d_C, size_t ldC,
                    const TestConfig& config) {
        size_t M = config.M;
        size_t N = config.N;
        size_t K = config.K;
        size_t warmup = config.warmup_runs;
        size_t bench = config.benchmark_runs;
    
        // Warmup
        for (size_t i = 0; i < warmup; ++i) {
            func(d_A, ldA, d_B, ldB, d_C, ldC, M, N, K);
        }
        // Benchmark
        Timer timer;
        timer.start();
        for (size_t i = 0; i < bench; ++i) {
            func(d_A, ldA, d_B, ldB, d_C, ldC, M, N, K);
        }
        double elapsed_ms = timer.stop();
        double avg_time = elapsed_ms / bench;

        return avg_time;
    }

    void print_result(const std::string& name, double time_ms, const TestConfig& config,
                    double max_error, bool is_correct, double ref_time_ms) {
        if (time_ms < 0) {
            std::cout << std::left << std::setw(25) << name << "FAILED" << std::endl;
            return;
        }
        size_t M = config.M;
        size_t N = config.N;
        size_t K = config.K;
        double ops = 2.0 * M * N * K; // GEMM operations
        double tflops = (ops / (time_ms * 1e-3)) / 1e12; // Convert to TFLOPS
        double bandwidth = ( (M * K + K * N + M * N) * sizeof(typeIn) / (time_ms * 1e-3) ) / 1e9; // GB/s
        double speedup = ref_time_ms / time_ms;
        
        std::cout << std::left << std::setw(25) << name
                    << std::fixed << std::setprecision(3) << std::setw(12) << time_ms
                    << std::setw(12) << tflops
                    << std::setw(15) << bandwidth
                    << std::setw(12) << (is_correct ? "PASS" : "FAIL")
                    << std::setw(16) << std::scientific << std::setprecision(2) << max_error 
                    << std::setw(16) << std::scientific << std::setprecision(2) << speedup 
                    << std::endl;
    }

    std::pair<bool, double> check_correctness(typeOut* C, typeOut* ref_C, size_t M, size_t N, double tolerance) {
        // Check correctness of the GEMM result
        double max_error = 0.0;
        for (size_t i = 0; i < M * N; ++i) {
            double error = std::abs(static_cast<double>(C[i]) - static_cast<double>(ref_C[i]));
            max_error = std::max(max_error, error);
        }
        return {max_error < tolerance, max_error};
    }

public:
    GEMMBenchmark(const std::vector<TestConfig> new_configs){
        configs = new_configs;
    };

    GEMMBenchmark(const std::vector<TestConfig> new_configs, std::vector<CustomGEMMFunction<typeIn, typeOut>> new_functions) {
        configs = new_configs;
        custom_functions = new_functions;
    };


    void add_custom_function(const std::string& name, 
                             std::function<void(typeIn*, size_t, typeIn*, size_t, typeOut*, size_t, size_t, size_t, size_t)> func) {
        custom_functions.emplace_back(name, func);
    }

    void add_custom_function(std::vector<CustomGEMMFunction<typeIn, typeOut>> new_functions) {
        custom_functions.insert(custom_functions.end(), new_functions.begin(), new_functions.end());
    }

    void add_custom_function(CustomGEMMFunction<typeIn, typeOut> func) {
        custom_functions.push_back(func);
    }

    void set_reference_function(std::function<void(typeIn*, size_t, typeIn*, size_t, typeOut*, size_t, size_t, size_t, size_t)> func) {
        reference_function = CustomGEMMFunction<typeIn, typeOut>("reference", func);
    }

    void set_reference_function(CustomGEMMFunction<typeIn, typeOut> ref_function) {
        reference_function = ref_function;
    }

    void set_reference_function() {
        reference_function = CustomGEMMFunction<typeIn, typeOut>("reference_cublas", reference_gemm_blas<typeIn, typeOut>);
    }

    void run_benchmark() {
        blas::blas_init();
        for (const auto& config : configs) {
            std::cout << "\n=== GEMM Benchmark ===" << std::endl;
            std::cout << "Problem size: M=" << config.M << ", N=" << config.N << ", K=" << config.K << std::endl;
            std::cout << "Precision: " << config.precision << std::endl;
            std::cout << "Warmup runs: " << config.warmup_runs << std::endl;
            std::cout << "Benchmark runs: " << config.benchmark_runs << std::endl;

            // Print header
            std::cout << std::string(90, '=') << std::endl;
            std::cout << std::left << std::setw(25) << "Function" 
                  << std::setw(12) << "Time(ms)"
                  << std::setw(12) << "TFLOPS"
                  << std::setw(15) << "Bandwidth(GB/s)"
                  << std::setw(12) << "Correctness"
                  << std::setw(16) << "Max Error" 
                  << std::setw(16) << "Speedup" 
                  << std::endl;
            std::cout << std::string(90, '-') << std::endl;
            size_t M = config.M;
            size_t N = config.N;
            size_t K = config.K;
            // int rand_seed = config.seed; // For future use if needed
            
            // Allocate memory and initialize data
            typeIn *h_A, *h_B;
            typeOut *h_C, *h_ref_C;
            typeIn *d_A, *d_B;
            typeOut *d_C, *d_ref_C;

            size_t ldA = K, ldB = N, ldC = N;  // Correct leading dimensions for row-major matrices
            h_A = new typeIn[M * K];
            h_B = new typeIn[K * N];
            h_C = new typeOut[M * N];
            h_ref_C = new typeOut[M * N];

            // Allocate device memory
            cudaMalloc(&d_A, M * K * sizeof(typeIn));
            cudaMalloc(&d_B, K * N * sizeof(typeIn));
            cudaMalloc(&d_C, M * N * sizeof(typeOut));
            cudaMalloc(&d_ref_C, M * N * sizeof(typeOut));

            // Initialize data 
            for (size_t i = 0; i < M * K; ++i) {
                float v = ((float)(i % 43) / 20.0 + 0.2 + (i % 19)); // Example initialization
                h_A[i] = static_cast<typeIn>(v); // Example initialization
            }
            for (size_t i = 0; i < K * N; ++i) {
                float v = ((float)(i % 43) / 20.0 + 0.2 + (i % 37)); // Example initialization
                h_B[i] = static_cast<typeIn>(v); // Example initialization
            }

            // Copy data to device
            cudaMemcpy(d_A, h_A, M * K * sizeof(typeIn), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, K * N * sizeof(typeIn), cudaMemcpyHostToDevice);

            
            cudaMemcpy(d_ref_C, h_ref_C, M * N * sizeof(typeOut), cudaMemcpyHostToDevice);
            double avg_time_ref = run_func(reference_function, d_A, ldA, d_B, ldB, d_ref_C, ldC, config);
            cudaMemcpy(h_ref_C, d_ref_C, M * N * sizeof(typeOut), cudaMemcpyDeviceToHost);
            print_result(reference_function.name, avg_time_ref, config, 0.0, true, avg_time_ref);
            
            for (const auto& func : custom_functions) {
                for (size_t i = 0; i < M * N; ++i) {
                    h_C[i] = static_cast<typeOut>(0); // Initialize output to zero
                }
                cudaMemcpy(d_C, h_C, M * N * sizeof(typeOut), cudaMemcpyHostToDevice);
                double avg_time = run_func(func, d_A, ldA, d_B, ldB, d_C, ldC, config);
                cudaMemcpy(h_C, d_C, M * N * sizeof(typeOut), cudaMemcpyDeviceToHost);

                bool is_correct = true;
                double max_error = 0.0;
                auto pair = check_correctness(h_C, h_ref_C, M, N, config.tolerance);
                is_correct = pair.first;
                max_error = pair.second;

                // Print results
                print_result(func.name, avg_time, config, max_error, is_correct, avg_time_ref);
            }
            std::cout << std::string(90, '=') << std::endl;
            // Clean up
            delete[] h_A;
            delete[] h_B;
            delete[] h_C;
            delete[] h_ref_C;
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            cudaFree(d_ref_C);
        }
        blas::blas_destroy();
    }
};
