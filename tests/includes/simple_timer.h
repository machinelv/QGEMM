#pragma once
#include <chrono>
#include <cuda_runtime.h>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    cudaEvent_t cuda_start, cuda_stop;
    bool use_cuda_events;

public:
    Timer(bool use_cuda = true) : use_cuda_events(use_cuda) {
        if (use_cuda_events) {
            cudaEventCreate(&cuda_start);
            cudaEventCreate(&cuda_stop);
        }
    }

    ~Timer() {
        if (use_cuda_events) {
            cudaEventDestroy(cuda_start);
            cudaEventDestroy(cuda_stop);
        }
    }

    void start() {
        if (use_cuda_events) {
            cudaEventRecord(cuda_start);
        } else {
            start_time = std::chrono::high_resolution_clock::now();
        }
    }

    double stop() {
        if (use_cuda_events) {
            cudaEventRecord(cuda_stop);
            cudaEventSynchronize(cuda_stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, cuda_start, cuda_stop);
            return static_cast<double>(milliseconds);
        } else {
            end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            return static_cast<double>(duration.count()) / 1000.0; // Convert to milliseconds
        }
    }
};
