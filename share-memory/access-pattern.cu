#include <iostream>
#include <cuda_runtime.h>

#define ROW 2048
#define COL 32

__global__ void setRowReadRow(float* out) {
  __shared__ float share[ROW][COL];
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int idx = y * COL + x;
  share[y][x] = idx;  // 32 * 32
  out[idx] = share[y][x]; 
}

__global__ void setColReadCol(float* out) {
  __shared__ float share[ROW][COL];
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int idx = y * COL + x;
  share[x][y] = idx;
  out[idx] = share[x][y];
}

__global__ void noShareMemory(float * out) {
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int idx = y * COL + x;
  out[idx] = idx;
}

int main(int argc, char* argv[]) {
  size_t bytes = sizeof(float) * ROW * COL;
  float* h_out = (float*)malloc(bytes);
  float* d_out;
  cudaEvent_t start;
  cudaEvent_t stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMalloc(&d_out, bytes);
  dim3 block(ROW, COL);
  dim3 grid(1, 1);
  cudaEventRecord(start);
  for (int i = 0; i < 100; ++i) {
    setRowReadRow<<<grid, block>>>(d_out);
  }
  cudaEventRecord(stop);
  cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << "setRowReadRow takes " << time / 100 << "\n";

  cudaEventRecord(start);
  for (int i = 0; i < 100; ++i) {
    setColReadCol<<<grid, block>>>(d_out);
  }
  cudaEventRecord(stop);
  cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaEventElapsedTime(&time, start, stop);
  std::cout << "setColReadCol takes " << time / 100 << "\n";

  cudaEventRecord(start);
  for (int i = 0; i < 100; ++i) {
    noShareMemory<<<grid, block>>>(d_out);
  }
  cudaEventRecord(stop);
  cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaEventElapsedTime(&time, start, stop);
  std::cout << "noShareMemory takes " << time / 100 << "\n";
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_out);
  free(h_out);
  return 0;
}