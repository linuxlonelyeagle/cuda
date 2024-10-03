#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>

#define FLOAT4(value)  *(float4*)(&(value))

__global__ void add_vector_base(float* a, float* b, float* c, int n) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t k = idx;
  if (k > n)
    return;
  c[k] = a[k] + b[k];
}

__global__ void add_vector_offset(float* a, float* b, float* c, int n, int offset) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t k = idx + offset;
  if (k > n)
    return;
  c[k] = a[k] + b[k];
}

__global__ void add_vector_float4(float* a, float* b, float* c, int n, int offset) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t k = idx * 4;
  if (k > n)
    return;
  float4 tmpA = FLOAT4(a[k]);
  float4 tmpB = FLOAT4(b[k]);
  float4 tmp;
  tmp.x = tmpA.x + tmpB.x;
  tmp.y = tmpA.y + tmpB.y;
  tmp.z = tmpA.z + tmpB.z;
  tmp.w = tmpA.w + tmpB.w;
  FLOAT4(a[k]) = tmp;
}

int main(int argc, char *argv[]) {
  int offset = 0;
  if (argc == 2)
    offset = atoi(argv[1]);

  size_t size = 1 << 16;
  size_t bytes = size * sizeof(float);
  size_t block = 512;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float time;
  float* h_a =(float*)malloc(bytes);
  float* h_b =(float*)malloc(bytes);
  float* h_c = (float*)malloc(bytes);
  float* d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  cudaEventRecord(start);
  add_vector_base<<<(size + block - 1) / block, block>>>(d_a, d_b, d_c, size);
  cudaEventRecord(stop);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&time, start, stop);
  printf("add_vector_base takes %f\n", time);

  cudaEventRecord(start);
  add_vector_offset<<<(size + block - 1) / block, block>>>(d_a, d_b, d_c, size, offset);
  cudaEventRecord(stop);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&time, start, stop);
  printf("add_vector_offset takes %f\n", time);

  cudaEventRecord(start);
  add_vector_float4<<<(size + block - 1) / block / 4, block>>>(d_a, d_b, d_c, size, offset);
  cudaEventRecord(stop);
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&time, start, stop);
  printf("add_vector_float4 takes %f\n", time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}
