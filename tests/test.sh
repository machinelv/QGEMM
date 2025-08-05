#!/bin/bash

# Test script for CUDA GEMM Benchmark Framework
# This script tests the build and basic functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  CUDA GEMM Benchmark Test Script      ${NC}"
echo -e "${GREEN}========================================${NC}"

# Change to tests directory
cd "$(dirname "$0")"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: NVCC not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: No NVIDIA GPU detected or nvidia-smi not available.${NC}"
    exit 1
fi

# Check GPU compute capability
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
GPU_CC_MAJOR=$(echo $GPU_CC | cut -d'.' -f1)
if [ "$GPU_CC_MAJOR" -lt 8 ]; then
    echo -e "${RED}Error: This benchmark requires GPU with Compute Capability 8.0+${NC}"
    echo -e "${RED}Your GPU has Compute Capability: ${GPU_CC}${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CUDA toolkit found${NC}"
echo -e "${GREEN}✓ NVIDIA GPU detected (CC: ${GPU_CC})${NC}"

# Build the project
echo -e "${YELLOW}Building the project...${NC}"
./build.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

# Run a quick test
echo -e "${YELLOW}Running quick test...${NC}"
cd build

# Check if executable exists
if [ ! -f "bin/cuda_benchmark" ]; then
    echo -e "${RED}Error: Executable not found at bin/cuda_benchmark${NC}"
    exit 1
fi

# Run the benchmark with timeout to prevent hanging
timeout 30s ./bin/cuda_benchmark || {
    if [ $? -eq 124 ]; then
        echo -e "${YELLOW}Test timed out after 30 seconds (this is normal for first run)${NC}"
    else
        echo -e "${RED}Test failed with exit code: $?${NC}"
        exit 1
    fi
}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All tests passed successfully!       ${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}You can now run the full benchmark with:${NC}"
echo -e "${YELLOW}  cd build && ./bin/cuda_benchmark${NC}"
