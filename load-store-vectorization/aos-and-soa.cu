#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

#define N (1 << 20)

struct innerStruct {
  float x;
  float y;
};

struct innerArray {
  float x[N];
  float y[N];
};

__global__ void testInnerStruct(innerStruct* input, innerStruct* output, int n) {
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (idx < n) {
    innerStruct tmp = input[idx];
    tmp.x += 1.0;
    tmp.y += 2.0;
    output[idx] = tmp;
  }
}

__global__ void testInnerArray(innerArray* input, innerArray* output, int n) {
  int idx = blockDim.x*blockIdx.x+threadIdx.x;
  if (idx < n) {
    output->x[idx] = input->x[idx] + 1.0;
    output->y[idx] = input->y[idx] + 2.0;
  }
}

__global__ void testInnerArrayHalf(innerArray* input, innerArray* output, int n) {
  int idx = blockDim.x*blockIdx.x+threadIdx.x;
  if (idx < n) {
    output->x[idx] = input->x[idx] + 1.0;
  }
}

int main(int argc, char* argv[]) {
  size_t blockSize = 512;
  size_t bytes = N * sizeof(innerStruct);
  cudaEvent_t start, stop;
  float msec;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  innerStruct* hInput0 =(innerStruct*)malloc(bytes);
  innerStruct* hOutput0 = (innerStruct*)malloc(bytes);
  for (int i = 0; i < N; ++i) {
    hInput0[i].x = 1;
    hInput0[i].y = 1;
  }
  innerStruct* dInput0, *dOutput0;
  cudaMalloc(&dInput0, bytes);
  cudaMalloc(&dOutput0, bytes);
  cudaMemcpy(dInput0, hInput0, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(start);
  testInnerStruct<<<(N + blockSize - 1) / blockSize, blockSize>>>(dInput0, dOutput0, N);
  cudaEventRecord(stop);
  cudaMemcpy(hOutput0, dOutput0, bytes, cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&msec, start, stop);
  printf("testInnerStruct takes %.3f msec\n", msec);
  cudaFree(dInput0);
  cudaFree(dOutput0);
  free(hInput0);
  free(hOutput0);

  // ----------------------------------------------------------------//
  innerArray* hInput = (innerArray*)malloc(sizeof(innerArray));
  innerArray* hOutput = (innerArray*)malloc(sizeof(innerArray));
  innerArray* dInput, *dOutput;
  cudaMalloc(&dInput, sizeof(innerArray));
  cudaMalloc(&dOutput, sizeof(innerArray));
  for (int i = 0; i < N; ++i) {
    hInput->x[i] = 1;
    hInput->y[i] = 1;
  }
  cudaMemcpy(dInput, hInput, sizeof(innerArray), cudaMemcpyHostToDevice);  
  cudaEventRecord(start);
  testInnerArray<<<(N + blockSize - 1) / blockSize, blockSize>>>(dInput, dOutput, N);
  cudaEventRecord(stop);
  cudaMemcpy(hOutput, dOutput, sizeof(innerArray), cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&msec, start, stop);
  printf("testInnerArray takes %.3f msec\n", msec);


  // ----------------------------------------------------------------//
  cudaEventRecord(start);
  testInnerArrayHalf<<<(N + blockSize - 1) / blockSize, blockSize>>>(dInput, dOutput, N);
  cudaEventRecord(stop);
  cudaMemcpy(hOutput, dOutput, sizeof(innerArray), cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&msec, start, stop);
  printf("testInnerArrayHalf takes %.3f msec\n", msec);

  cudaFree(dInput);
  cudaFree(dOutput);
  free(hInput);
  free(hOutput);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}