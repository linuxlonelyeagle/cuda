#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void add_vector(float* a, float* b, float* c, int n, int offset) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t k = idx + offset;
  if (idx < n || k < n)
    return;
  c[n] = a[k] + b[k];
}

int main(int argc, char *argv[]) {
  int offset = 0;
  if (argc == 2)
    offset = atoi(argv[1]);

  size_t size = 1 << 14;
  size_t bytes = size * sizeof(float);
  size_t block = 512;
  float* h_a =(float*)malloc(bytes);
  float* h_b =(float*)malloc(bytes);
  float* h_c = (float*)malloc(bytes);
  float* d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  add_vector<<<(size + block - 1) / block, block>>>(d_a, d_b, d_c, size, offset);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
