// Example configuration for different test scenarios

#pragma once

#include "test_config.h"

// Small matrices for quick testing
std::vector<TestConfig> get_small_test_configs(const std::string& precision) {
    return {
        TestConfig(128, 128, 128, precision, 3, 10),
        TestConfig(256, 256, 256, precision, 3, 10),
        TestConfig(512, 512, 512, precision, 3, 10)
    };
}

// Medium matrices for regular benchmarking
std::vector<TestConfig> get_medium_test_configs(const std::string& precision) {
    return {
        TestConfig(512, 512, 512, precision, 5, 20),
        TestConfig(1024, 1024, 1024, precision, 5, 20),
        TestConfig(2048, 2048, 2048, precision, 5, 20)
    };
}

// Large matrices for stress testing
std::vector<TestConfig> get_large_test_configs(const std::string& precision) {
    return {
        TestConfig(2048, 2048, 2048, precision, 3, 10),
        TestConfig(4096, 4096, 4096, precision, 3, 10),
        TestConfig(8192, 8192, 8192, precision, 3, 5)  // Fewer runs for very large matrices
    };
}

// Mixed sizes for comprehensive testing
std::vector<TestConfig> get_mixed_test_configs(const std::string& precision) {
    return {
        // Square matrices
        TestConfig(512, 512, 512, precision),
        TestConfig(1024, 1024, 1024, precision),
        
        // Tall matrices
        TestConfig(2048, 512, 1024, precision),
        TestConfig(4096, 1024, 2048, precision),
        
        // Wide matrices  
        TestConfig(512, 2048, 1024, precision),
        TestConfig(1024, 4096, 2048, precision),
        
        // Common neural network sizes
        TestConfig(1024, 4096, 1024, precision),  // Transformer FFN
        TestConfig(4096, 1024, 4096, precision),  // Transformer projection
    };
}
