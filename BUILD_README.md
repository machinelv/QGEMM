# QGEMM - Quantized GEMM Kernels

A high-performance quantized GEMM (General Matrix Multiplication) library supporting AMD CDNA3 and NVIDIA SM80/SM89 architectures.

## Features

- **Multi-architecture support**: AMD CDNA3 (gfx942) and NVIDIA SM80/SM89
- **Python integration**: Compiled kernels can be called from Python
- **Dynamic library**: Can be compiled as shared library (.so)
- **Testing framework**: Includes C++ test programs
- **CMake build system**: Modern CMake with automatic GPU detection

## Supported Architectures

### AMD
- **CDNA3**: gfx942

### NVIDIA
- **SM80**: A100, DGX A100
- **SM89**: RTX 4090, L40, L4

## Quick Start

### Prerequisites

**For AMD GPUs:**
```bash
# ROCm installation required
sudo apt install rocm-dev hip-dev
```

**For NVIDIA GPUs:**
```bash
# CUDA Toolkit installation required
sudo apt install nvidia-cuda-toolkit
```

**Common dependencies:**
```bash
# PyTorch (with appropriate GPU support)
pip install torch torchvision torchaudio

# Build tools
sudo apt install cmake build-essential python3-dev
```

### Building

#### Method 1: Using the build script (Recommended)

```bash
# Auto-detect GPU and build
./build.sh

# Build for NVIDIA with tests
./build.sh --vendor=NVIDIA --arch=80,89 --tests

# Build for AMD CDNA3
./build.sh --vendor=AMD --arch=gfx942 --tests

# Clean build
./build.sh --clean --tests
```

#### Method 2: Using CMake directly

```bash
mkdir build && cd build

# For NVIDIA
cmake .. -DTARGET_VENDOR=NVIDIA -DTARGET_GPU_ARCH="80;89" -DBUILD_TESTS=ON

# For AMD
cmake .. -DTARGET_VENDOR=AMD -DTARGET_GPU_ARCH=gfx942 -DBUILD_TESTS=ON

make -j$(nproc)
```

#### Method 3: Using Python setup

```bash
# Install as Python package
pip install .

# Development install
pip install -e .
```

### Output Files

After successful build, you'll find:

```
build/
├── lib/
│   ├── libqgemm.so          # Main GEMM library
│   └── qgemm_python.so      # Python module
└── bin/
    └── qgemm_test           # Test executable (if BUILD_TESTS=ON)
```

## Usage

### C++ Usage

```cpp
#include "gemm.h"

// Use GEMM functions
// Implementation depends on your kernel interfaces
```

### Python Usage

```python
import sys
sys.path.append('build/lib')
import qgemm_python

# Use the compiled GEMM kernels
# Implementation depends on your Python bindings
```

### Running Tests

```bash
# Run C++ tests
./build/bin/qgemm_test

# Run Python tests
python tests/gemm_csrc.py
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `TARGET_VENDOR` | Auto-detect | Target GPU vendor (AMD/NVIDIA) |
| `TARGET_GPU_ARCH` | Vendor-specific | Target GPU architecture |
| `BUILD_TESTS` | OFF | Build test programs |
| `BUILD_SHARED_LIBS` | ON | Build shared libraries |

## Architecture-Specific Settings

### AMD CDNA3 (gfx942)
- Uses HIP for GPU programming
- Supports composable_kernel integration
- Optimized for MI300 series

### NVIDIA SM80/SM89
- Uses CUDA for GPU programming
- Supports multiple SM architectures in single binary
- Optimized for A100 and RTX 4090 series

## Project Structure

```
QGEMM/
├── CMakeLists.txt           # Main build configuration
├── build.sh                 # Build script
├── setup.py                 # Python package setup
├── include/                 # Header files
│   ├── gemm.h
│   ├── gpu_lib.h
│   ├── gpu_types.h
│   └── timer.h
├── src/                     # Source code
│   ├── gemm/               # GEMM kernels
│   │   └── sm80_gemm_kernel.cpp
│   └── mix_gemm/           # Mixed precision GEMM
│       ├── hip_mix_gemm_kernel.cpp
│       └── sm80_mix_gemm_kernel.cpp
├── tests/                   # Test files
│   ├── benchmark.cpp
│   ├── benchmark.ipynb
│   └── gemm_csrc.py
└── triton/                 # Triton implementations
    └── fp8_gemm.py
```

## Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure proper drivers are installed
2. **CMake errors**: Check if all dependencies are installed
3. **Compilation errors**: Verify target architecture matches your GPU

### Build Logs

CMake will print a configuration summary:
```
=== QGEMM Configuration Summary ===
Target Vendor: NVIDIA
Target GPU Architecture: 80;89
Build Tests: ON
Build Shared Libraries: ON
Build Directory: /path/to/build
====================================
```

## Contributing

1. Ensure your code builds on both AMD and NVIDIA platforms
2. Add appropriate tests for new kernels
3. Update documentation for new features

## License

[Add your license information here]
