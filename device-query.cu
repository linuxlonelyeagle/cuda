#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main(int argc, char* argv[]) {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  cout << deviceProp.name << endl;
  cout << "global memory: " << deviceProp.totalGlobalMem << " bytes" << endl;  
  cout << "global memory: " << deviceProp.totalGlobalMem / 1024 / 1024 << " MB" << endl;
  cout << "Multiprocessor(SM) count: " << deviceProp.multiProcessorCount << endl;
  cout << "l2 cache: " << deviceProp.l2CacheSize << " bytes" << endl;
  cout << "l2 cache: " << deviceProp.l2CacheSize / 1024 / 1024 << "MB" << endl;
  cout << "Memory Bus Width: " << deviceProp.memoryBusWidth <<  " bit" << endl;
  cout << "Total amount of shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << endl;
  cout << "Total shared memory per multiprocessor: " << deviceProp.sharedMemPerMultiprocessor << " bytes" << endl;
  cout << "Total shared memory per multiprocessor: " << deviceProp.sharedMemPerMultiprocessor / 1024 << " KB" << endl;
  cout << "warp size: " << deviceProp.warpSize << endl;
  return 0;
}
