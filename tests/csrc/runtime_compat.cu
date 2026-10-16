#include <cuda_fp8.h>
#include <cuda_runtime.h>

__global__ void forwarding_kernel(__grid_constant__ cudaStream_t stream) {
  (void)stream;
  __nv_fp8_e4m3 value{};
  (void)value;
}

void compile_forwarding_names() {
  cudaError_t error = cudaGetLastError();
  (void)cudaGetErrorString(error);
  (void)cudaLaunchKernel;
  (void)cudaMemcpyAsync;
}
