#!/bin/bash

# QGEMM Build Script
# Usage: ./build.sh [OPTIONS]
# 
# Options:
#   --vendor=AMD|NVIDIA     Target vendor (default: auto-detect)
#   --arch=ARCH             Target GPU architecture 
#                          AMD: gfx942 (CDNA3)
#                          NVIDIA: 80,89 (SM80,SM89)
#   --tests                 Build test programs
#   --test-kernels          Build test kernel files (files starting with test_)
#   --clean                 Clean build directory before building
#   --help                  Show this help message

set -e

# Default values
VENDOR=""
ARCH=""
BUILD_TESTS="OFF"
BUILD_TEST_KERNELS="OFF"
CLEAN_BUILD=false
BUILD_DIR="build"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vendor=*)
            VENDOR="${1#*=}"
            shift
            ;;
        --arch=*)
            ARCH="${1#*=}"
            shift
            ;;
        --tests)
            BUILD_TESTS="ON"
            shift
            ;;
        --test-kernels)
            BUILD_TEST_KERNELS="ON"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help)
            echo "QGEMM Build Script"
            echo ""
            echo "Usage: ./build.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --vendor=AMD|NVIDIA     Target vendor (default: auto-detect)"
            echo "  --arch=ARCH             Target GPU architecture"
            echo "                          AMD: gfx942 (CDNA3)"
            echo "                          NVIDIA: 80,89 (SM80,SM89)"
            echo "  --tests                 Build test programs"
            echo "  --test-kernels          Build test kernel files (files starting with test_)"
            echo "  --clean                 Clean build directory before building"
            echo "  --help                  Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./build.sh --vendor=NVIDIA --arch=80,89 --tests"
            echo "  ./build.sh --vendor=AMD --arch=gfx942 --test-kernels"
            echo "  ./build.sh --clean --tests --test-kernels"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf $BUILD_DIR
    exit 0
fi

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Prepare CMake arguments
CMAKE_ARGS="-DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=$BUILD_TESTS -DBUILD_TEST_KERNELS=$BUILD_TEST_KERNELS"

if [ ! -z "$VENDOR" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DTARGET_VENDOR=$VENDOR"
fi

if [ ! -z "$ARCH" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DTARGET_GPU_ARCH=$ARCH"
fi

echo "Configuring with CMake..."
echo "Arguments: $CMAKE_ARGS"

# Configure with CMake
cmake .. $CMAKE_ARGS

echo "Building..."
make -j

echo ""
echo "Build completed successfully!"
echo "Output files are in: $(pwd)/"
echo ""
echo "Libraries:"
echo "  - libqgemm.so (main GEMM library)"
echo "  - qgemm_python.so (Python module)"
if [ "$BUILD_TESTS" = "ON" ]; then
    echo "Tests:"
    echo "  - qgemm_test (C++ test executable)"
fi
if [ "$BUILD_TEST_KERNELS" = "ON" ]; then
    echo "Test Kernels:"
    echo "  - libqgemm_test_kernels.so (test kernel library)"
    echo "  - qgemm_test_kernels_python.so (Python test kernel module)"
fi
echo ""
echo "To install, run: make install"
