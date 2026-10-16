#pragma once

#include <musa_runtime.h>

#ifndef __grid_constant__
#define __grid_constant__
#endif

using cudaError_t = musaError_t;
using cudaStream_t = musaStream_t;
using cudaLaunchConfig_t = musaLaunchConfig_t;
using cudaLaunchAttribute = musaLaunchAttribute;

#ifndef cudaSuccess
#define cudaSuccess musaSuccess
#endif
#ifndef cudaStreamPerThread
#define cudaStreamPerThread musaStreamPerThread
#endif
#ifndef cudaGetErrorString
#define cudaGetErrorString musaGetErrorString
#endif
#ifndef cudaGetLastError
#define cudaGetLastError musaGetLastError
#endif
#ifndef cudaLaunchKernel
#define cudaLaunchKernel musaLaunchKernel
#endif
#ifndef cudaLaunchKernelEx
#define cudaLaunchKernelEx musaLaunchKernelEx
#endif
#ifndef cudaMemcpy
#define cudaMemcpy musaMemcpy
#endif
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync musaMemcpyAsync
#endif
#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#endif
#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#endif
#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute musaDeviceGetAttribute
#endif
#ifndef cudaGetDevice
#define cudaGetDevice musaGetDevice
#endif
