#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <random>
#include <iomanip>
#include <cmath>
#include <memory>

#include <cuda_runtime.h>

#include "timer.h"
#include "gemm.h"

// Test configuration for SM80 architecture
struct TestConfig {
    size_t M, N, K;
    std::string precision;
    size_t warmup_runs;
    size_t benchmark_runs;
    int seed; // For reproducibility
    double tolerance;

    TestConfig(size_t m, size_t n, size_t k, const std::string& prec,
               size_t warmup = 5, size_t bench = 20, double tol = 1e-2, int s = 42)
        : M(m), N(n), K(k), precision(prec), warmup_runs(warmup),
          benchmark_runs(bench), tolerance(tol), seed(s){}
};

template<typename typeIn, typename typeOut>
struct CustomGEMMFunction {
    std::string name;
    std::function<void(typeIn*, size_t, typeIn*, size_t, typeOut*, size_t, size_t, size_t, size_t)> func;

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
    std::vector<std::pair<std::string, 
        CustomGEMMFunction<typeIn, typeIn, typeOut>>> custom_functions;
    std::vector<TestConfig> configs;

    /**
    auto allocate_memory(const TestConfig& config) {
        size_t size_A = config.M * config.K;
        size_t size_B = config.K * config.N;
        size_t size_C = config.M * config.N;
        
        typeIn *d_A, *h_A;
        typeIn *d_B, *h_B;
        typeOut *d_C, *d_ref, *h_C, *h_ref;
        
        // Allocate host memory
        h_A = new typeIn[size_A];
        h_B = new typeIn[size_B];
        h_C = new typeOut[size_C];
        h_ref = new typeOut[size_C];

        // Allocate device memory
        cudaMalloc(&d_A, size_A * sizeof(typeIn));
        cudaMalloc(&d_B, size_B * sizeof(typeIn));
        cudaMalloc(&d_C, size_C * sizeof(typeOut));
        cudaMalloc(&d_ref, size_C * sizeof(typeOut));
        
        return std::make_tuple(d_A, d_B, d_C, d_ref, h_A, h_B, h_C);
    }
    
    void initialize_data(typeIn* h_A, typeIn* h_B, typeOut* h_C, const TestConfig& config) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if constexpr (std::is_same_v<typeIn, int8_t>) {
            std::uniform_int_distribution<int> dis(-127, 127);
            for (size_t i = 0; i < config.M * config.K; ++i) {
                h_A[i] = static_cast<typeIn>(dis(gen));
            }
            for (size_t i = 0; i < config.K * config.N; ++i) {
                h_B[i] = static_cast<typeIn>(dis(gen));
            }
        } else {
            std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
            for (size_t i = 0; i < config.M * config.K; ++i) {
                h_A[i] = static_cast<typeIn>(dis(gen));
            }
            for (size_t i = 0; i < config.K * config.N; ++i) {
                h_B[i] = static_cast<typeIn>(dis(gen));
            }
        }
        
        // Initialize C to zero
        for (size_t i = 0; i < config.M * config.N; ++i) {
            h_C[i] = static_cast<typeOut>(0);
        }
    }
    
    void copy_to_device(typeIn* d_A, typeIn* d_B, typeOut* d_C, 
                       typeIn* h_A, typeIn* h_B, typeOut* h_C, 
                       const TestConfig& config) {
        cudaMemcpy(d_A, h_A, config.M * config.K * sizeof(typeIn), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, config.K * config.N * sizeof(typeIn), cudaMemcpyHostToDevice);     
        cudaMemcpy(d_C, h_C, config.M * config.N * sizeof(typeOut), cudaMemcpyHostToDevice);
    }
    

    std::pair<double, bool> check_correctness(ElementC* result, ElementC* reference, const TestConfig& config) {
        size_t size = config.M * config.N;
        ElementC* h_result = new ElementC[size];
        ElementC* h_reference = new ElementC[size];
        
        cudaMemcpy(h_result, result, size * sizeof(ElementC), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_reference, reference, size * sizeof(ElementC), cudaMemcpyDeviceToHost);
        
        double max_error = 0.0;
        bool is_correct = true;
        
        for (size_t i = 0; i < size; ++i) {
            double diff = std::abs(static_cast<double>(h_result[i]) - static_cast<double>(h_reference[i]));
            double ref_val = std::abs(static_cast<double>(h_reference[i]));
            double relative_error = ref_val > 1e-10 ? diff / ref_val : diff;
            
            max_error = std::max(max_error, relative_error);
            
            if (relative_error > config.tolerance) {
                is_correct = false;
            }
        }
        
        delete[] h_result;
        delete[] h_reference;
        
        return {max_error, is_correct};
    }
    
    void print_result(const std::string& name, double time_ms, const TestConfig& config, 
                     double max_error, bool is_correct) {
        if (time_ms < 0) {
            std::cout << std::left << std::setw(25) << name << "FAILED" << std::endl;
            return;
        }
        
        // Calculate TFLOPS
        double ops = 2.0 * config.M * config.N * config.K; // GEMM operations
        double tflops = (ops / (time_ms * 1e-3)) / 1e12;
        
        // Calculate bandwidth (rough estimate)
        size_t bytes = (config.M * config.K + config.K * config.N + config.M * config.N) * sizeof(ElementA);
        double bandwidth = (bytes / (time_ms * 1e-3)) / 1e9;
        
        std::cout << std::left << std::setw(25) << name
                  << std::fixed << std::setprecision(3) << std::setw(12) << time_ms
                  << std::setw(12) << tflops
                  << std::setw(15) << bandwidth
                  << std::setw(12) << (is_correct ? "PASS" : "FAIL")
                  << std::scientific << std::setprecision(2) << max_error << std::endl;
    }

    void cleanup_memory(typeIn* d_A, typeIn* d_B, typeOut* d_C, typeOut* d_D, typeOut* d_ref,
                       typeIn* h_A, typeIn* h_B, typeOut* h_C) {
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_D);
        cudaFree(d_ref);
    }
    */
public:
    GEMMBenchmark(const std::vector<TestConfig>& new_configs) : configs(new_configs) {};

    GEMMBenchmark(const std::vector<TestConfig>& new_configs, 
        std::vector<std::pair<std::string, 
        std::function<void(typeIn*, size_t, typeIn*, size_t, typeOut*, size_t, size_t, size_t)>>> new_functions) : configs(new_configs), custom_functions(new_functions) {};


    void reference_gemm(typeIn* A, size_t ldA,
                        typeIn* B, size_t ldB, 
                        typeOut* C, size_t ldC,
                        size_t M, size_t N, size_t K) {
        // Call the reference GEMM implementation
        cutlass_gemm_test<typeIn, typeOut>(A, ldA, B, ldB, C, ldC, M, N, K);
    }

    void add_custom_function(const std::string& name, 
                             std::function<void(typeIn*, size_t, typeIn*, size_t, typeOut*, size_t, size_t, size_t)> func) {
        custom_functions.emplace_back(name, func);
    }

    void add_custom_function(std::vector<std::pair<std::string, 
        std::function<void(typeIn*, size_t, typeIn*, size_t, typeOut*, size_t, size_t, size_t)>>> new_functions) {
        custom_functions.insert(custom_functions.end(), new_functions.begin(), new_functions.end());
    }

    bool check_correctness(typeOut* C, typeOut* ref_C, size_t M, size_t N, double tolerance) {
        // Check correctness of the GEMM result
        double max_error = 0.0;
        #pragma omp parallel for reduction(max:max_error)
        for (size_t i = 0; i < M * N; ++i) {
            double error = std::abs(static_cast<double>(C[i]) - static_cast<double>(ref_C[i]));
            max_error = std::max(max_error, error);
        }
        return max_error < tolerance;
    }

    void run_benchmark() {
        std::cout << "\n=== GEMM Benchmark ===" << std::endl;
        std::cout << "Problem size: M=" << config.M << ", N=" << config.N << ", K=" << config.K << std::endl;
        std::cout << "Precision: " << config.precision << std::endl;
        std::cout << "Warmup runs: " << config.warmup_runs << std::endl;
        std::cout << "Benchmark runs: " << config.benchmark_runs << std::endl;
        std::cout << std::string(90, '=') << std::endl;
        
        // Print header
        std::cout << std::left << std::setw(25) << "Function" 
                  << std::setw(12) << "Time(ms)"
                  << std::setw(12) << "TFLOPS"
                  << std::setw(15) << "Bandwidth(GB/s)"
                  << std::setw(12) << "Correctness"
                  << std::setw(15) << "Max Error" << std::endl;
        std::cout << std::string(90, '-') << std::endl;

        for (const auto& config : configs) {
            size_t M = config.M;
            size_t N = config.N;
            size_t K = config.K;
            int rand_seed = config.seed;
            
            // Allocate memory and initialize data
            typeIn *h_A, *h_B, *h_C, *h_ref_C;
            typeIn *d_A, *d_B, *d_C, *d_ref_C;

            size_t ldA = M, ldB = K, ldC = M;
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
                h_A[i] = static_cast<typeIn>(rand()); // Example initialization
            }
            for (size_t i = 0; i < K * N; ++i) {
                h_B[i] = static_cast<typeIn>(rand()); // Example initialization
            }
            for (size_t i = 0; i < M * N; ++i) {
                h_C[i] = static_cast<typeOut>(0); // Initialize output to zero
            }
            // Copy data to device
            cudaMemcpy(d_A, h_A, M * K * sizeof(typeIn), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, K * N * sizeof(typeIn), cudaMemcpyHostToDevice);
            cudaMemcpy(d_C, h_C, M * N * sizeof(typeOut), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ref_C, h_ref_C, M * N * sizeof(typeOut), cudaMemcpyHostToDevice);

            for (const auto& func : custom_functions) {
                // Warmup
                for (size_t i = 0; i < config.warmup_runs; ++i) {
                    func(d_A, ldA, d_B, ldB, d_C, ldC, M, N, K);
                }
                // Benchmark
                Timer timer;
                timer.start();
                for (size_t i = 0; i < config.benchmark_runs; ++i) {
                    func(d_A, ldA, d_B, ldB, d_C, ldC, M, N, K);
                }
                double elapsed_ms = timer.stop();
                double avg_time = elapsed_ms / config.benchmark_runs;

                // Calculate TFLOPS
                double tflops = (2.0 * M * N * K) / (avg_time * 1e9); // TFLOPS = 2 * M * N * K * 1e-12 

                // Check correctness                
                double max_error = 0.0; // Placeholder for max error calculation
                bool is_correct = true; // Placeholder for correctness check
                typeOut* ref_C = new typeOut[M * N];
                reference_gemm(d_A, ldA, d_B, ldB, ref_C, ldC, M, N, K);

                cudaMemcpy(h_C, d_C, M * N * sizeof(typeOut), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_ref_C, d_ref_C, M * N * sizeof(typeOut), cudaMemcpyDeviceToHost);

                is_correct = check_correctness(h_C, h_ref_C, M, N, config.tolerance);
                max_error = 0.0; // Calculate max error if needed

                if (!is_correct) {
                    std::cerr << "Error: GEMM function " << func.name << " failed correctness check." << std::endl;
                } else {
                    // Print results
                    print_result(func.name, avg_time, config, max_error, is_correct);
                }
            }

            // Clean up
            delete[] A;
            delete[] B;
            delete[] C;
        }

    }
};