#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__device__ float computeAdd(float a, float b) {
  float result;
  asm ("add.f32 %0,%1,%2;" :  "=f"(result) : 
                               "f"(a),
                               "f"(b));
  return result;                              
}

__global__ void kern(float* input, float a, float b) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  input[index] = computeAdd(a, b);
}

int main(int argc, char *argv[]) {
  int size = 1 << 10;
  cout << "size: " << size << "\n";
  int blockSize = 256;
  if (argc > 1) {
    blockSize = atoi(argv[1]);
  }
  dim3 block(blockSize, 1);
  dim3 grid((size - 1) / block.x + 1, 1);
  cout << "grid: " << grid.x << endl;
  cout << "block:" << block.x << endl;
  size_t bytes = size * sizeof(float);
  
  // alloc host memory.
  float *outputDataHost = (float*)malloc(bytes);
  for (int i = 0; i < size; ++i) {
    outputDataHost[i] = 0;
  }

  // alloc device memory.
  float *outputDataDevice = nullptr;
  cudaMalloc((void**)&outputDataDevice, bytes);
  cudaMemcpy(outputDataDevice, outputDataHost, bytes, cudaMemcpyHostToDevice);
  kern<<<grid, block>>>(outputDataDevice, 1.5, 2.0);
  cudaMemcpy(outputDataHost, outputDataDevice, bytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < size; ++i) {
    cout << outputDataHost[i] << " ";
  }
  cout << "\n";
  return 0;
}
