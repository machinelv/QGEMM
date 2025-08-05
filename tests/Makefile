# Makefile for CUDA GEMM Benchmark Framework
# Alternative build system to CMake

# Compiler settings
NVCC := nvcc
CXX := g++

# CUDA architectures
CUDA_ARCH := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89

# Directories
SRC_DIR := ../src
INCLUDE_DIR := ../include
TEST_INCLUDE_DIR := includes
BUILD_DIR := build
BIN_DIR := $(BUILD_DIR)/bin
OBJ_DIR := $(BUILD_DIR)/obj

# Source files
GEMM_SRC := $(SRC_DIR)/gemm/test_cutlass_gemm.cpp
MAIN_SRC := cuda_benchmark.cpp

# Object files
GEMM_OBJ := $(OBJ_DIR)/test_cutlass_gemm.o
MAIN_OBJ := $(OBJ_DIR)/cuda_benchmark.o

# Output executable
TARGET := $(BIN_DIR)/cuda_benchmark

# Include paths
INCLUDES := -I$(INCLUDE_DIR) -I$(TEST_INCLUDE_DIR) -I$(SRC_DIR)/include

# Try to find CUTLASS
CUTLASS_PATHS := ../ref/cutlass/include /usr/local/include/cutlass /opt/cutlass/include
CUTLASS_INCLUDE := $(foreach path,$(CUTLASS_PATHS),$(if $(wildcard $(path)/cutlass/cutlass.h),-I$(path)))

ifeq ($(CUTLASS_INCLUDE),)
    $(error CUTLASS not found. Please install CUTLASS or set CUTLASS_INCLUDE manually)
endif

INCLUDES += $(CUTLASS_INCLUDE)

# Compiler flags
NVCC_FLAGS := -std=c++17 $(CUDA_ARCH) -O3 --expt-relaxed-constexpr -DTEST_ON_CUDA
CXX_FLAGS := -std=c++17 -O3 -Wall -Wextra -DTEST_ON_CUDA

# CUDA libraries
CUDA_LIBS := -lcudart -lcublas

# Default target
all: $(TARGET)

# Create directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BIN_DIR)
	mkdir -p $(OBJ_DIR)

# Compile GEMM source
$(GEMM_OBJ): $(GEMM_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile main source
$(MAIN_OBJ): $(MAIN_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Link executable
$(TARGET): $(GEMM_OBJ) $(MAIN_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(CUDA_LIBS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

# Run the benchmark
run: $(TARGET)
	cd $(BUILD_DIR) && ./bin/cuda_benchmark

# Test build (compile only)
test-build: $(TARGET)
	@echo "Build successful!"

# Show compiler information
info:
	@echo "NVCC version:"
	@$(NVCC) --version
	@echo ""
	@echo "CUTLASS include path: $(CUTLASS_INCLUDE)"
	@echo "Include paths: $(INCLUDES)"
	@echo "CUDA architectures: $(CUDA_ARCH)"

# Install (copy to system directories)
install: $(TARGET)
	@echo "Installing benchmark to /usr/local/bin/"
	sudo cp $(TARGET) /usr/local/bin/cuda_gemm_benchmark

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build the benchmark (default)"
	@echo "  clean      - Clean build files"
	@echo "  run        - Build and run the benchmark"
	@echo "  test-build - Test compilation only"
	@echo "  info       - Show compiler information"
	@echo "  install    - Install to system directory"
	@echo "  help       - Show this help message"

.PHONY: all clean run test-build info install help
