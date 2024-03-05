#include "cuda_runtime.h"
#include <cstdio>

#define N 32

__global__ void kernel(int *a, int *b, int *c) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

int main(int argc, char *argv[]) {
  int *host_a = nullptr, *host_b = nullptr, *host_c = nullptr;
  int *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  int dataSize = N * sizeof(int);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // allocate some page-locked memory
  cudaHostAlloc((void **)&host_a, dataSize, cudaHostAllocDefault);
  cudaHostAlloc((void **)&host_b, dataSize, cudaHostAllocDefault);
  cudaHostAlloc((void **)&host_c, dataSize, cudaHostAllocDefault);
  for (int i = 0; i < N; ++i) {
    host_a[i] = i;
    host_b[i] = i;
  }

  // alloc gpu memrey.
  cudaMalloc((void **)&dev_a, dataSize);
  cudaMalloc((void **)&dev_b, dataSize);
  cudaMalloc((void **)&dev_c, dataSize);

  // copy host memrey to gpu memrey.
  cudaMemcpyAsync(dev_a, host_a, dataSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(dev_b, host_b, dataSize, cudaMemcpyHostToDevice, stream);

  // asynchronous calls to copy over the data to the GPU, we can asynchronously
  // call our kernel. This call will return control to the host immediately and
  // the kernel will run at some point after the previous operation in the
  // stream is complete.
  kernel<<<1, N, 0, stream>>>(dev_a, dev_b, dev_c);

  // copy result.
  cudaMemcpyAsync(host_c, dev_c, dataSize, cudaMemcpyDeviceToHost, stream);
  // Blocks until stream has completed all operations.
  cudaStreamSynchronize(stream);

  // print.
  printf("print A:\n");
  for (int i = 0; i < N; ++i) {
    printf("%d ", host_a[i]);
  }
  printf("\nprint B:\n");
  for (int i = 0; i < N; ++i) {
    printf("%d ", host_b[i]);
  }
  printf("\nprint C:\n");
  for (int i = 0; i < N; ++i) {
    printf("%d ", host_c[i]);
  }
  printf("\n");
  return 0;
}
