cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21

nclude(config.cmake)
if(TARGET_VENDOR STREQUAL "AMD")
set(GPU_TARGETS ${TARGET_GPU_ARCH} CACHE STRING "GPU architectures to build for HIP")
set(ENV{PYTORCH_ROCM_ARCH} ${TARGET_GPU_ARCH})
endif()
set(CMAKE_CXX_STANDARD 20)
cmake_policy(VERSION 3.21.3...3.27)

if(TARGET_VENDOR STREQUAL "AMD")
project(QGEMM LANGUAGES HIP CXX)
else()
project(QGEMM LANGUAGES CXX CUDA)
endif()

find_package(Torch REQUIRED PATHS ${LIBTORCH_DIR})

include_directories(include)
include_directories(${LIBTORCH_DIR}/include)


if(TARGET_VENDOR STREQUAL "AMD")
    find_package(hip)
    find_package(composable_kernel COMPONENTS device_other_operations device_gemm_operations)
    add_compile_definitions(TEST_ON_HIP)
    add_compile_definitions(HIP_ENABLE_WARP_SYNC_BUILTINS)
else()
    add_compile_definitions(TEST_ON_CUDA)
endif()

