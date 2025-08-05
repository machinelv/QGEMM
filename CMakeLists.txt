cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21

# Set build output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib)

# Project configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
cmake_policy(VERSION 3.21.3...3.27)

# Options
option(BUILD_TESTS "Build test programs" OFF)
option(BUILD_TEST_KERNELS "Build test kernel files (files starting with test_)" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(TARGET_VENDOR "Target vendor (AMD/NVIDIA)" "NVIDIA")
option(TARGET_GPU_ARCH "Target GPU architecture" "")

# Auto-detect GPU vendor if not specified
if(NOT TARGET_VENDOR)
    find_program(ROCM_SMI "rocm-smi")
    find_program(NVIDIA_SMI "nvidia-smi")
    
    if(ROCM_SMI)
        set(TARGET_VENDOR "AMD")
        message(STATUS "Auto-detected AMD GPU")
    elseif(NVIDIA_SMI)
        set(TARGET_VENDOR "NVIDIA")
        message(STATUS "Auto-detected NVIDIA GPU")
    else()
        set(TARGET_VENDOR "NVIDIA")
        message(WARNING "Could not detect GPU vendor, defaulting to NVIDIA")
    endif()
endif()

# Set GPU architectures based on vendor
if(TARGET_VENDOR STREQUAL "AMD")
    if(NOT TARGET_GPU_ARCH)
        set(TARGET_GPU_ARCH "gfx942" CACHE STRING "Default CDNA3 architecture")
    endif()
    set(GPU_TARGETS ${TARGET_GPU_ARCH} CACHE STRING "GPU architectures to build for HIP")
    set(ENV{PYTORCH_ROCM_ARCH} ${TARGET_GPU_ARCH})
    project(QGEMM LANGUAGES HIP CXX)
else()
    if(NOT TARGET_GPU_ARCH)
        set(TARGET_GPU_ARCH "80;89" CACHE STRING "Default NVIDIA architectures")
    endif()
    project(QGEMM LANGUAGES CXX CUDA)
endif()

# Find required packages
set(Torch_DIR "/data00/home/lyuqizheng/APPs/miniconda3/envs/mytorch/lib/python3.13/site-packages/torch/share/cmake/Torch")
list(APPEND CMAKE_PREFIX_PATH ${Torch_DIR})
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(Torch CONFIG REQUIRED)

# Set include directories
include_directories(include)
include_directories(src/include)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

# Platform-specific setup
if(TARGET_VENDOR STREQUAL "AMD")
    find_package(hip REQUIRED)
    find_package(composable_kernel COMPONENTS device_other_operations device_gemm_operations)
    add_compile_definitions(TEST_ON_HIP)
    add_compile_definitions(HIP_ENABLE_WARP_SYNC_BUILTINS)
    set(GPU_COMPILE_FLAGS "--offload-arch=${TARGET_GPU_ARCH}")
else()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(TEST_ON_CUDA)
    
    # Set CUDA architectures
    string(REPLACE ";" "," CUDA_ARCH_LIST "${TARGET_GPU_ARCH}")
    set(CMAKE_CUDA_ARCHITECTURES ${TARGET_GPU_ARCH})
    set(CUDA_COMPILE_FLAGS "-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89")
endif()

# Collect source files
set(GEMM_SOURCES "")
set(TEST_SOURCES "")

if(TARGET_VENDOR STREQUAL "AMD")
    if(BUILD_TEST_KERNELS)
        # AMD HIP test sources
        file(GLOB HIP_TEST_SOURCES "src/**/test_*.cpp")
        list(APPEND TEST_SOURCES ${HIP_TEST_SOURCES})
    else()
        # AMD HIP sources
        file(GLOB HIP_GEMM_SOURCES "src/mix_gemm/hip_*.cpp")
        list(APPEND GEMM_SOURCES ${HIP_GEMM_SOURCES})
    endif()
else()
    if(BUILD_TEST_KERNELS)
        # NVIDIA CUDA test sources
        file(GLOB CUDA_TEST_SOURCES "src/**/test_*.cpp")
        list(APPEND TEST_SOURCES ${CUDA_TEST_SOURCES})
    else()
        # NVIDIA CUDA sources
        file(GLOB CUDA_GEMM_SOURCES "src/**/sm*_*.cpp" "src/**/sm*_*.cpp")
        list(APPEND GEMM_SOURCES ${CUDA_GEMM_SOURCES})
    endif()
endif()


if(BUILD_LIBS)
# Create GEMM library
add_library(qgemm ${GEMM_SOURCES})

# Set target properties
set_target_properties(qgemm PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Link libraries
target_link_libraries(qgemm 
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
)

if(TARGET_VENDOR STREQUAL "AMD")
    target_link_libraries(qgemm hip::host)
    if(composable_kernel_FOUND)
        target_link_libraries(qgemm composable_kernel::device_other_operations composable_kernel::device_gemm_operations)
    endif()
    # Set HIP compilation flags
    set_target_properties(qgemm PROPERTIES
        HIP_SEPARABLE_COMPILATION ON
    )
else()
    target_link_libraries(qgemm CUDA::cudart CUDA::cublas)
    # Set CUDA compilation flags
    set_target_properties(qgemm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Compiler-specific flags
if(TARGET_VENDOR STREQUAL "AMD")
    target_compile_options(qgemm PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:${GPU_COMPILE_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
    )
else()
    target_compile_options(qgemm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
    )
endif()

# Build tests if enabled
if(BUILD_TESTS)
    add_executable(qgemm_test tests/benchmark.cpp)
    
    set_target_properties(qgemm_test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    
    target_link_libraries(qgemm_test qgemm ${TORCH_LIBRARIES})
    
    if(TARGET_VENDOR STREQUAL "AMD")
        target_link_libraries(qgemm_test hip::host)
    else()
        target_link_libraries(qgemm_test CUDA::cudart CUDA::cublas)
    endif()
endif()

# Build test kernels if enabled
if(BUILD_TEST_KERNELS AND TEST_SOURCES)
    # Create test kernel library
    add_library(qgemm_test_kernels ${TEST_SOURCES})
    
    set_target_properties(qgemm_test_kernels PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    target_include_directories(qgemm_test_kernels PRIVATE 
        include
        ${TORCH_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    
    target_link_libraries(qgemm_test_kernels 
        ${TORCH_LIBRARIES}
        ${Python3_LIBRARIES}
    )
    
    if(TARGET_VENDOR STREQUAL "AMD")
        target_link_libraries(qgemm_test_kernels hip::host)
        if(composable_kernel_FOUND)
            target_link_libraries(qgemm_test_kernels composable_kernel::device_other_operations composable_kernel::device_gemm_operations)
        endif()
        set_target_properties(qgemm_test_kernels PROPERTIES
            HIP_SEPARABLE_COMPILATION ON
        )
        target_compile_options(qgemm_test_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:${GPU_COMPILE_FLAGS}>
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
        )
    else()
        target_link_libraries(qgemm_test_kernels CUDA::cudart CUDA::cublas)
        # Find CUTLASS for test kernels
        find_package(PkgConfig QUIET)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass")
            set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass")
            target_include_directories(qgemm_test_kernels PRIVATE "${CUTLASS_DIR}/include")
            target_include_directories(qgemm_test_kernels PRIVATE "${CUTLASS_DIR}/tools/util/include")
        endif()
        
        set_target_properties(qgemm_test_kernels PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        target_compile_options(qgemm_test_kernels PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_FLAGS}>
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
        )
    endif()
    
    # Create Python module for test kernels
    add_library(qgemm_test_kernels_python MODULE ${TEST_SOURCES})
    
    set_target_properties(qgemm_test_kernels_python PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    target_include_directories(qgemm_test_kernels_python PRIVATE 
        include
        ${TORCH_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
    )
    
    target_link_libraries(qgemm_test_kernels_python 
        ${TORCH_LIBRARIES}
        ${Python3_LIBRARIES}
    )
    
    if(TARGET_VENDOR STREQUAL "AMD")
        target_link_libraries(qgemm_test_kernels_python hip::host)
        if(composable_kernel_FOUND)
            target_link_libraries(qgemm_test_kernels_python composable_kernel::device_other_operations composable_kernel::device_gemm_operations)
        endif()
        set_target_properties(qgemm_test_kernels_python PROPERTIES
            HIP_SEPARABLE_COMPILATION ON
        )
        target_compile_options(qgemm_test_kernels_python PRIVATE
            $<$<COMPILE_LANGUAGE:HIP>:${GPU_COMPILE_FLAGS}>
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
        )
    else()
        target_link_libraries(qgemm_test_kernels_python CUDA::cudart CUDA::cublas)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass")
            target_include_directories(qgemm_test_kernels_python PRIVATE "${CUTLASS_DIR}/include")
            target_include_directories(qgemm_test_kernels_python PRIVATE "${CUTLASS_DIR}/tools/util/include")
        endif()
        
        set_target_properties(qgemm_test_kernels_python PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
        target_compile_options(qgemm_test_kernels_python PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_FLAGS}>
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
        )
    endif()
endif()


# Python binding target (for pybind11 integration)
add_library(qgemm_python MODULE ${GEMM_SOURCES})

set_target_properties(qgemm_python PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(qgemm_python 
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
)

if(TARGET_VENDOR STREQUAL "AMD")
    target_link_libraries(qgemm_python hip::host)
    if(composable_kernel_FOUND)
        target_link_libraries(qgemm_python composable_kernel::device_other_operations composable_kernel::device_gemm_operations)
    endif()
    set_target_properties(qgemm_python PROPERTIES
        HIP_SEPARABLE_COMPILATION ON
    )
else()
    target_link_libraries(qgemm_python CUDA::cudart CUDA::cublas)
    set_target_properties(qgemm_python PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# Apply same compilation flags to Python module
if(TARGET_VENDOR STREQUAL "AMD")
    target_compile_options(qgemm_python PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:${GPU_COMPILE_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
    )
else()
    target_compile_options(qgemm_python PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_COMPILE_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC>
    )
endif()

# Install targets
install(TARGETS qgemm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(TARGETS qgemm_python
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

if(BUILD_TESTS)
    install(TARGETS qgemm_test
        RUNTIME DESTINATION bin
    )
endif()

if(BUILD_TEST_KERNELS AND TEST_SOURCES)
    install(TARGETS qgemm_test_kernels
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
    
    install(TARGETS qgemm_test_kernels_python
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

endif() # BUILD_LIBS

# Print configuration summary
message(STATUS "=== QGEMM Configuration Summary ===")
message(STATUS "Target Vendor: ${TARGET_VENDOR}")
message(STATUS "Target GPU Architecture: ${TARGET_GPU_ARCH}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "Build Test Kernels: ${BUILD_TEST_KERNELS}")
message(STATUS "Build Shared Libraries: ${BUILD_SHARED_LIBS}")
message(STATUS "Build Directory: ${CMAKE_BINARY_DIR}/build")
message(STATUS "GEMM Sources: ${GEMM_SOURCES}")
if(BUILD_TEST_KERNELS)
    message(STATUS "Test Sources: ${TEST_SOURCES}")
endif()
message(STATUS "====================================")