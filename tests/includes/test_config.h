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


    */


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

    void reference_gemm(typeIn* A, size_t ldA,
                        typeIn* B, size_t ldB, 
                        typeOut* C, size_t ldC,
                        size_t M, size_t N, size_t K) {
        
    }

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

    void run_benchmark() {
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
                h_A[i] = static_cast<typeIn>((i % 43) / 20.0); // Example initialization
            }
            for (size_t i = 0; i < K * N; ++i) {
                h_B[i] = static_cast<typeIn>((i % 37) / 18.0); // Example initialization
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

    }
};
