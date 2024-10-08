#include <cuda_runtime.h>
#include <iostream>

// read row, write col;
__global__ void transpose0(float *in, float *out, int nx, int ny) {
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  out[iy + ix * ny] = in[ix + iy * nx];
}

// read col, write row;
__global__ void transpose1(float *in, float *out, int nx, int ny) {
  int ix = blockDim.x * blockIdx.x + threadIdx.x;
  int iy = blockDim.y * blockIdx.y + threadIdx.y;
  out[ix + iy * nx] = in[iy + ix * ny];
}

__global__ void transposeOptimized(float *input, float *output, int m, int n) {
  int colID_input = threadIdx.x + blockDim.x * blockIdx.x;
  int rowID_input = threadIdx.y + blockDim.y * blockIdx.y;
  __shared__ float sdata[32][33];
  if (colID_input < m && rowID_input < n) {
    int index_input = colID_input + rowID_input * n;
    sdata[threadIdx.y][threadIdx.x] = input[index_input];
    __syncthreads();
    int dst_col = threadIdx.x + blockIdx.y * blockDim.y;
    int dst_row = threadIdx.y + blockIdx.x * blockDim.x;
    output[dst_col + dst_row * m] = sdata[threadIdx.x][threadIdx.y];
  }
}

int main(int argc, char *argv[]) {
  int nx = 1 << 12;
  int ny = 1 << 12;
  size_t bytes = nx * ny * sizeof(float);
  float *h_in = (float *)malloc(bytes);
  float *h_out = (float *)malloc(bytes);
  float *d_in, *d_out;
  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  // init data;
  for (int i = 0; i < nx * ny; ++i) {
    h_in[i] = i;
  }
  cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
  cudaEvent_t start;
  cudaEvent_t stop;
  float msec;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  {
    dim3 block(32, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
      transpose0<<<grid, block>>>(d_in, d_out, nx, ny);
    }
    cudaEventRecord(stop);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&msec, start, stop);
    printf("read row, write col\n");
    printf("transpose0 takes %.3f msec\n", msec / 100);
  }
  {
    dim3 block(32, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
      transpose1<<<grid, block>>>(d_in, d_out, nx, ny);
    }
    cudaEventRecord(stop);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&msec, start, stop);
    printf("read col, write row\n");
    printf("transpose1 takes %.3f msec\n", msec / 100);
  }
  {
    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
      transposeOptimized<<<grid, block>>>(d_in, d_out, nx, ny);
    }
    cudaEventRecord(stop);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&msec, start, stop);
    printf("transposeOptimized takes %.3f msec\n", msec / 100);
    /*
    for (int i = 0; i < ny; ++i) {
      for (int j = 0; j < nx; ++j) {
        int idx = i * nx + j;
        printf("%f ", h_out[idx]);
      }
      printf("\n");
    }
    printf("\n");
    */
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_out);
  free(h_in);
  free(h_out);
  return 0;
}
