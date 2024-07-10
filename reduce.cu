#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>
using namespace std;

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

__global__ void reduce0(int* globalInputData, int* globalOutputData, size_t size) {
  extern __shared__ int shareData[];
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  shareData[tid] = globalInputData[idx];
  __syncthreads();
  
  // do reducetion in share memery.
  // Problem: highly divergent 
  // warps are very inefficient, and 
  // % operator is very slow
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      shareData[tid] += shareData[tid + stride];
    }
    __syncthreads();
  }
  // write result for this block to global mem;
  if (tid == 0) 
    globalOutputData[blockIdx.x] = shareData[0];
}

// New Problem: Shared Memory Bank Conflicts
__global__ void reduce1(int *globalInput, int *globalOutput, size_t size) {
  extern __shared__ int shareData[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size)
    return;
  shareData[tid] = globalInput[i];
  __syncthreads();
  
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = 2 * stride * tid;
    if (index < blockDim.x) {
      shareData[index] += shareData[index + stride];
    }
    __syncthreads();
  }
  if (tid == 0) 
    globalOutput[blockIdx.x] = shareData[0];
}

// Problem: Half of the threads are idle on first loop iteration!This is wasteful...
__global__ void reduce2(int *globalInput, int* globalOutput, size_t size) {
  extern __shared__ int shareData[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > size)
    return;
  shareData[tid] = globalInput[i];
  __syncthreads();
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shareData[tid] += shareData[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0)
    globalOutput[blockIdx.x] = shareData[0];
}

__global__ void reduce3(int *globalInput, int* globalOutput, size_t size) {
  extern __shared__ int shareData[];
  unsigned int tid = threadIdx.x;

  // block 0 : read(0, 1)
  // block 1 : read(2, 3) 
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  shareData[tid] = globalInput[i] + globalInput[i + blockDim.x];

  __syncthreads();
  for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shareData[tid] += shareData[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0)
    globalOutput[blockIdx.x] = shareData[0];
}


__device__ void warpReduce(volatile int* shareData, int tid) {
  shareData[tid] += shareData[tid + 32];
  shareData[tid] += shareData[tid + 16];
  shareData[tid] += shareData[tid + 8];
  shareData[tid] += shareData[tid + 4];
  shareData[tid] += shareData[tid + 2];
  shareData[tid] += shareData[tid + 1];
}

// Unrolling the Last Warp
__global__ void reduce4(int *globalInput, int* globalOutput, size_t size) {
  extern __shared__ int shareData[];
  unsigned int tid = threadIdx.x;

  // block 0 : read(0, 1)
  // block 1 : read(2, 3) 
  unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  shareData[tid] = globalInput[i] + globalInput[i + blockDim.x];

  __syncthreads();
  for (size_t stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      shareData[tid] += shareData[tid + stride];
    }
    __syncthreads();
  }
  if (tid < 32) 
    warpReduce(shareData, tid);    
  if (tid == 0)
    globalOutput[blockIdx.x] = shareData[0];
}

int main(int argc, char* argv[]) {
  int size = 1 << 24;
  cout << "array size: " << size << endl;
  int blockSize = 1024;
  if (argc > 1) {
    blockSize = atoi(argv[1]);
  }
  dim3 block(blockSize, 1);
  dim3 grid((size - 1) / block.x + 1, 1);
  cout << "grid: " << grid.x << endl;
  cout << "block:" << block.x << endl;

  // alloc host memory
  size_t bytes = size * sizeof(int);
  int *inputDataHost = (int*)malloc(bytes);
  int *outputDataHost = (int*)malloc(grid.x * sizeof(int));
  for (int i = 0; i < size; ++i)
    inputDataHost[i] = 1;

  // alloc device memory
  int *inputDataDevice = nullptr;
  int *outputDataDevice = nullptr;
  cudaMalloc((void**)&inputDataDevice, bytes);
  cudaMalloc((void**)&outputDataDevice, grid.x * sizeof(int));

  cudaMemcpy(inputDataDevice, inputDataHost, bytes, cudaMemcpyHostToDevice);
  double start = cpuSecond();
  // indicate the size() of shared memory in the lunch function.
  reduce0<<<grid, block, blockSize * sizeof(int)>>>(inputDataDevice, outputDataDevice, size);
  double end = cpuSecond() - start;
  cout << "reduce0: " << end << endl; 
  cudaMemcpy(outputDataHost, outputDataDevice, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < grid.x; ++i) 
    outputDataHost[i] = 0;
  
  // test reduce1
  start = cpuSecond();
  reduce1<<<grid, block, blockSize * sizeof(int)>>>(inputDataDevice, outputDataDevice, size);
  end = cpuSecond() - start;
  cout << "reduce1: " << end << endl; 
  cudaMemcpy(outputDataHost, outputDataDevice, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < grid.x; ++i)
    outputDataHost[i] = 0;

  // test reduce2
  start = cpuSecond();
  reduce2<<<grid, block, blockSize * sizeof(int)>>>(inputDataDevice, outputDataDevice, size);
  end = cpuSecond() - start;
  cout << "reduce2: " << end << endl;
  cudaMemcpy(outputDataHost, outputDataDevice, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < grid.x; ++i)
    outputDataHost[i] = 0;

  // test reduce3
  start = cpuSecond();
  reduce3<<<grid, block, blockSize * sizeof(int)>>>(inputDataDevice, outputDataDevice, size);
  end = cpuSecond() - start;
  cout << "reduce3: " << end << endl;
  cudaMemcpy(outputDataHost, outputDataDevice, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  for (int i = 0; i < grid.x; ++i)
    outputDataHost[i] = 0;

  // test reduce4
  start = cpuSecond();
  reduce4<<<grid, block, blockSize * sizeof(int)>>>(inputDataDevice, outputDataDevice, size);
  end = cpuSecond() - start;
  cout << "reduce4: " << end << endl;
  cudaMemcpy(outputDataHost, outputDataDevice, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  return 0;
}