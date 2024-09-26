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

/*
__global__ void reduceUnrollWarp8(int * g_idata,int * g_odata,unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*8+threadIdx.x;
	//convert global data pointer to the
	int *idata = g_idata + blockIdx.x*blockDim.x*8;
	//unrolling 8;
	if(idx+7 * blockDim.x<n)
	{
		int a1=g_idata[idx];
		int a2=g_idata[idx+blockDim.x];
		int a3=g_idata[idx+2*blockDim.x];
		int a4=g_idata[idx+3*blockDim.x];
		int a5=g_idata[idx+4*blockDim.x];
		int a6=g_idata[idx+5*blockDim.x];
		int a7=g_idata[idx+6*blockDim.x];
		int a8=g_idata[idx+7*blockDim.x];
		g_idata[idx]=a1+a2+a3+a4+a5+a6+a7+a8;

	}
	__syncthreads();
	//in-place reduction in global memory
	for (int stride = blockDim.x/2; stride>32; stride >>=1)
	{
		if (tid <stride)
		{
			idata[tid] += idata[tid + stride];
		}
		//synchronize within block
		__syncthreads();
	}
	//write result for this block to global mem
	if(tid<32)
	{
		volatile int *vmem = idata;
		vmem[tid]+=vmem[tid+32];
		vmem[tid]+=vmem[tid+16];
		vmem[tid]+=vmem[tid+8];
		vmem[tid]+=vmem[tid+4];
		vmem[tid]+=vmem[tid+2];
		vmem[tid]+=vmem[tid+1];

	}

	if (tid == 0)
		g_odata[blockIdx.x] = 1;

}

__global__ void reduceUnrollWarp18 (int* gIdata, int* gOdata, unsigned int n) {
  unsigned int tid = threadIdx.x;
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  int* idata = blockIdx.x * blockDim.x * 8 + gIdata;
  // unrolling 8
  if (idx + 7 * blockDim.x < n) {
    int a = idata[idx];
    int b = idata[idx +     blockDim.x];
    int c = idata[idx + 2 * blockDim.x];
    int d = idata[idx + 3 * blockDim.x];
    int e = idata[idx + 4 * blockDim.x];
    int f = idata[idx + 5 * blockDim.x];
    int g = idata[idx + 6 * blockDim.x];
    int h = idata[idx + 7 * blockDim.x];
    gIdata[idx] = a + b + c + d + e + f + g + h;
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride)
      idata[tid] += idata[tid + stride];
    __syncthreads();
  }
  if (tid < 32) {
    volatile int *vmem = idata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }
  if (tid == 0) {
    gOdata[blockIdx.x] = idata[0];
  }
}
*/
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

__global__ void reduceGmem(int *g_idata, int *g_odata, unsigned int n) {
  unsigned tid = threadIdx.x;
  int* idata = g_idata + blockDim.x * blockIdx.x;
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n)
    return;
  if (blockDim.x >= 1024 && tid < 512)
    idata[tid] = idata[tid] + idata[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256)
    idata[tid] = idata[tid] + idata[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128)
    idata[tid] = idata[tid] + idata[tid + 128];
  if (blockDim.x >= 128 && tid < 64)
    idata[tid] = idata[tid] + idata[tid + 64];
  if (tid < 32) {
    volatile int* tmp = idata;
    tmp[tid] = tmp[tid] + tmp[tid + 32];
    tmp[tid] = tmp[tid] + tmp[tid + 16];
    tmp[tid] = tmp[tid] + tmp[tid + 8];
    tmp[tid] = tmp[tid] + tmp[tid + 4];
    tmp[tid] = tmp[tid] + tmp[tid + 2];
    tmp[tid] = tmp[tid] + tmp[tid + 1];
  }
  if (tid == 0)
    g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceSmem(int *g_idata, int *g_odata, unsigned int n) {
  unsigned tid = threadIdx.x;
  int* idata = g_idata + blockDim.x * blockIdx.x;
  unsigned idx = blockDim.x * blockIdx.x + threadIdx.x;
  extern __shared__ int share[];
  share[tid] = idata[tid];
  __syncthreads();
  if (idx < n)
    return;
  if (blockDim.x >= 1024 && tid < 512)
    share[tid] = share[tid] + share[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256)
    share[tid] = share[tid] + share[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128)
    share[tid] = share[tid] + share[tid + 128];
  if (blockDim.x >= 128 && tid < 64)
    share[tid] = share[tid] + share[tid + 64];
  if (tid < 32) {
    volatile int* tmp = share;
    tmp[tid] = tmp[tid] + tmp[tid + 32];
    tmp[tid] = tmp[tid] + tmp[tid + 16];
    tmp[tid] = tmp[tid] + tmp[tid + 8];
    tmp[tid] = tmp[tid] + tmp[tid + 4];
    tmp[tid] = tmp[tid] + tmp[tid + 2];
    tmp[tid] = tmp[tid] + tmp[tid + 1];
  }
  if (tid == 0)
    g_odata[blockIdx.x] = share[0];
}

__global__ void reduceSmemUnroll(int* g_idata, int* g_odata, unsigned int n) {
  unsigned tid = threadIdx.x;
  unsigned idx = blockDim.x * blockIdx.x * 4 + threadIdx.x;
  extern __shared__ int share[];
  int t = 0;
  if (idx + 3 * blockDim.x <= n) {
    int a = g_idata[idx];
    int b = g_idata[idx + blockDim.x];
    int c = g_idata[idx + 2 * blockDim.x];
    int d = g_idata[idx + 3 * blockDim.x];
    t = a + b + c + d;
  }
  share[tid] = t;
  __syncthreads();
  if (blockDim.x >= 1024 && tid < 512)
    share[tid] = share[tid] + share[tid + 512];
  __syncthreads();
  if (blockDim.x >= 512 && tid < 256)
    share[tid] = share[tid] + share[tid + 256];
  __syncthreads();
  if (blockDim.x >= 256 && tid < 128)
    share[tid] = share[tid] + share[tid + 128];
  if (blockDim.x >= 128 && tid < 64)
    share[tid] = share[tid] + share[tid + 64];
  if (tid < 32) {
    volatile int* tmp = share;
    tmp[tid] = tmp[tid] + tmp[tid + 32];
    tmp[tid] = tmp[tid] + tmp[tid + 16];
    tmp[tid] = tmp[tid] + tmp[tid + 8];
    tmp[tid] = tmp[tid] + tmp[tid + 4];
    tmp[tid] = tmp[tid] + tmp[tid + 2];
    tmp[tid] = tmp[tid] + tmp[tid + 1];
  }
  if (tid == 0)
    g_odata[blockIdx.x] = share[0];
}

int main(int argc, char* argv[]) {
  int size = 1 << 24;
  cout << "array size: " << size << endl;
  int blockSize = 1024;
  if (argc > 1) {
    blockSize = atoi(argv[1]);
  }
  dim3 block(blockSize, 1);
  dim3 grid((size + block.x - 1) / block.x, 1);
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
  /*
  cudaFree(outputDataDevice);
  
  // test reduceUnrollWarp8
  // size = 1024 
  cudaMalloc(&outputDataDevice, (grid.x / 8 * sizeof(int)));
  start = cpuSecond();
  reduceUnrollWarp8<<<grid.x / 8, block>>>(inputDataDevice, outputDataDevice, size);
  end = cpuSecond() - start;
  cudaMemcpy(outputDataHost, outputDataDevice, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost);
  cout << "reduceUnrollWarp8: " << end << endl;
  
  for (int i = 0; i < grid.x / 8; ++i) {
    cout << outputDataHost[i] << " ";
  }
  */

  // test reduceGmem
  start = cpuSecond();
  reduceGmem<<<grid, block>>>(inputDataDevice, outputDataDevice, size);
  end = cpuSecond() - start;
  cout << "reduceGmem: " << end << endl;
  cudaMemcpy(outputDataHost, outputDataDevice, grid.x * sizeof(int), cudaMemcpyDeviceToHost);

  // test reduceSmem
  start = cpuSecond();
  reduceSmem<<<grid, block, block.x * sizeof(int)>>>(inputDataDevice, outputDataDevice, size);
  end = cpuSecond() - start;
  cout << "reduceSmem: " << end << endl;
  cudaMemcpy(outputDataHost, outputDataDevice, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  
  // test reduceSmemUnroll
  start = cpuSecond();
  reduceSmemUnroll<<<grid.x / 4, block, block.x * sizeof(int)>>>(inputDataDevice, outputDataDevice, size);
  end = cpuSecond() - start;
  cout << "reduceSmemUnroll: " << end << endl;
  cudaMemcpy(outputDataHost, outputDataDevice, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
  free(inputDataHost);
  free(outputDataHost);
  cudaFree(inputDataDevice);
  cudaFree(outputDataDevice);
  return 0;
}
