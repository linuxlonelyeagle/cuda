#include <iostream>
#include <cuda_runtime.h>

__global__ void sumMatrix(float* a, float* b, float* c, int nx, int ny) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= nx || y >= ny)
    return;
  int idx = y * nx + x;
  c[idx] = b[idx] + a[idx];
}

int main(int argc, char* argv[]) {
  int nx = 1 << 14;
  int ny = 1 << 14;
  size_t bytes = nx * ny * sizeof(float);
  cudaEvent_t start, stop;
  float msec;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float* h_a = (float*)malloc(bytes);
  float* h_b = (float*)malloc(bytes);
  float* h_c = (float*)malloc(bytes);
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      int idx = i * nx + j;
      h_a[idx] = 1.0;
      h_b[idx] = 2.0;
    }
  }
  float* d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
  int dimx = 32;
  int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + dimx - 1) / dimx, (ny + dimy -1) / dimy);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
      sumMatrix<<<grid, block>>>(d_a, d_b, d_c, nx, ny);
    }
    cudaEventRecord(stop);
    cudaEventElapsedTime(&msec, start, stop);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);  
    printf("sumMatrix takes %f msec\n", msec);

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