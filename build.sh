#!/bin/bash

# CUDA GEMM Benchmark Build Script
# This script builds the CUDA GEMM benchmark test framework

set -e  # Exit on any error

GPU_ARCH=$1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  CUDA GEMM Benchmark Build Script     ${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo -e "${RED}Error: CMakeLists.txt not found. Please run this script from the tests directory.${NC}"
    exit 1
fi

# Check CUDA installation
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: NVCC not found. Please ensure CUDA is properly installed.${NC}"
    exit 1
fi

echo -e "${YELLOW}CUDA version:${NC}"
nvcc --version

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Removing existing build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo -e "${YELLOW}Configuring project with CMake...${NC}"
cmake .. \
    -DUSE_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build the project
echo -e "${YELLOW}Building project...${NC}"
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Build completed successfully!        ${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo -e "${YELLOW}Executable location: ${BUILD_DIR}/bin/cuda_benchmark${NC}"
    echo -e "${YELLOW}To run the benchmark:${NC}"
    echo -e "${YELLOW}  cd ${BUILD_DIR}${NC}"
    echo -e "${YELLOW}  ./bin/cuda_benchmark${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  Build failed!                        ${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
