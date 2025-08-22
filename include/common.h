#pragma once

#if defined(__CUDACC__)
#ifndef USE_CUDA
#define USE_CUDA
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#ifndef CUDA_ARCH
#define CUDA_ARCH 80
#endif
#elif  defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 890
#ifndef CUDA_ARCH
#define CUDA_ARCH 89
#endif
#endif

#include <cuda_runtime.h>


#elif defined(__HIPCC__)
#ifndef USE_HIP
#define USE_HIP
#endif

#include <hip/hip_runtime.h>

#endif

#include <iostream>
