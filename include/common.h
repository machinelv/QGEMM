#pragma once

#if defined(__CUDACC__)
#ifndef TEST_ON_CUDA
#define TEST_ON_CUDA
#endif

#include <cuda_runtime.h>


#elif defined(__HIPCC__)
#ifndef TEST_ON_HIP
#define TEST_ON_HIP
#endif

#include <hip/hip_runtime.h>

#endif

#include <iostream>
