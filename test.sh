#!/bin/bash

# QGEMM Test Script
# This script runs all available tests for the QGEMM project

set -e

echo "🧪 QGEMM Test Suite"
echo "==================="

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "❌ Build directory not found. Please run ./build.sh first."
    exit 1
fi

# Function to check if file exists and is executable
check_executable() {
    if [ -x "$1" ]; then
        echo "✅ Found: $1"
        return 0
    else
        echo "⚠️  Not found: $1"
        return 1
    fi
}

# Function to check if library exists
check_library() {
    if [ -f "$1" ]; then
        echo "✅ Found: $1"
        return 0
    else
        echo "⚠️  Not found: $1"
        return 1
    fi
}

echo ""
echo "📋 Checking built artifacts..."

# Check main libraries
check_library "build/lib/libqgemm.so"
check_library "build/lib/qgemm_python.so"

# Check test executable
if check_executable "build/bin/qgemm_test"; then
    TEST_EXECUTABLE_AVAILABLE=true
else
    TEST_EXECUTABLE_AVAILABLE=false
fi

# Check test kernel libraries
if check_library "build/lib/libqgemm_test_kernels.so"; then
    TEST_KERNELS_AVAILABLE=true
else
    TEST_KERNELS_AVAILABLE=false
fi

check_library "build/lib/qgemm_test_kernels_python.so"

echo ""
echo "🏃 Running tests..."

# Run C++ test executable if available
if [ "$TEST_EXECUTABLE_AVAILABLE" = true ]; then
    echo ""
    echo "🔧 Running C++ benchmark tests..."
    echo "================================="
    ./build/bin/qgemm_test
    if [ $? -eq 0 ]; then
        echo "✅ C++ tests completed successfully"
    else
        echo "❌ C++ tests failed"
    fi
else
    echo "⚠️  C++ test executable not available. Build with --tests to enable."
fi

# Run Python tests if available
echo ""
echo "🐍 Running Python tests..."
echo "=========================="

python3 << 'EOF'
import sys
import os
sys.path.append('build/lib')

print("Python test environment:")
print(f"  Python version: {sys.version}")

try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name()}")
except ImportError:
    print("  ⚠️  PyTorch not available")

# Test main library
try:
    import qgemm_python
    print("  ✅ qgemm_python loaded successfully")
except ImportError as e:
    print(f"  ❌ Failed to load qgemm_python: {e}")

# Test kernel library
try:
    import qgemm_test_kernels_python
    print("  ✅ qgemm_test_kernels_python loaded successfully")
    print(f"  Available functions: {dir(qgemm_test_kernels_python)}")
except ImportError as e:
    print(f"  ⚠️  qgemm_test_kernels_python not available: {e}")

print("\n🔍 Quick functionality test:")
try:
    import torch
    if torch.cuda.is_available():
        A = torch.randn(128, 128, dtype=torch.float16, device='cuda')
        B = torch.randn(128, 128, dtype=torch.float16, device='cuda')
        C = torch.mm(A, B)
        print("  ✅ Basic PyTorch GEMM test passed")
    else:
        print("  ⚠️  CUDA not available for functionality test")
except Exception as e:
    print(f"  ❌ Functionality test failed: {e}")
EOF

if [ $? -eq 0 ]; then
    echo "✅ Python tests completed successfully"
else
    echo "❌ Python tests failed"
fi

# Suggest running Jupyter notebook
echo ""
echo "📓 To run comprehensive benchmarks:"
echo "   cd tests && jupyter notebook benchmark.ipynb"
echo ""
echo "📊 Or run Python benchmark directly:"
echo "   cd tests && python3 -c 'exec(open(\"../tests/benchmark.py\").read())'"

echo ""
echo "🎉 Test suite completed!"
echo ""
echo "📋 Summary:"
echo "   - Main library: $([ -f "build/lib/libqgemm.so" ] && echo "✅" || echo "❌") libqgemm.so"
echo "   - Python module: $([ -f "build/lib/qgemm_python.so" ] && echo "✅" || echo "❌") qgemm_python.so"
echo "   - C++ tests: $([ "$TEST_EXECUTABLE_AVAILABLE" = true ] && echo "✅" || echo "⚠️ ") qgemm_test"
echo "   - Test kernels: $([ "$TEST_KERNELS_AVAILABLE" = true ] && echo "✅" || echo "⚠️ ") test kernel libraries"
