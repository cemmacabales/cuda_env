#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_world() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Hello, World! Thread %d\n", tid);
}

int main() {
  std::cout << "Starting CUDA Hello World program...\n";
  std::cout << "Launching kernel with 1 block and 10 threads\n";
  
  hello_world<<<1, 10>>>();
  
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cout << "CUDA not available. Running CPU fallback version:\n";
    // CPU fallback - simulate what the kernel would do
    for (int tid = 0; tid < 10; tid++) {
      printf("Hello, World! Thread %d\n", tid);
    }
  }
  
  std::cout << "Kernel execution completed.\n";
  
  return 0;
}