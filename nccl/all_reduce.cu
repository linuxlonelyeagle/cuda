#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

int main(int argc, char* argv[]) {
  int size = 32, bytes = size * sizeof(float);
  int myRank, nRanks, localRank = 0;

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));


  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  ncclUniqueId id;
  ncclComm_t comm;
  float *devSendBuff, *devRecvBuff;
  float *hostSendBuff, *hostRecvBuff;
  cudaStream_t s;

    //get NCCL unique ID at rank 0 and broadcast it to all others
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // picking a GPU based on localRank, allocate device buffers
  CUDACHECK(cudaSetDevice(localRank));
  CUDACHECK(cudaMalloc(&devSendBuff, bytes));
  CUDACHECK(cudaMalloc(&devRecvBuff, bytes));
  
  // alloc host memrey and init host memrey.
  CUDACHECK(cudaHostAlloc(&hostSendBuff, bytes, cudaHostAllocDefault));
  CUDACHECK(cudaHostAlloc(&hostRecvBuff, bytes, cudaHostAllocDefault));
  for (int i = 0; i < size; ++i) 
    hostSendBuff[i] = (float)i;
  
  CUDACHECK(cudaStreamCreate(&s));
  CUDACHECK(cudaMemcpyAsync(devSendBuff, hostSendBuff, bytes, cudaMemcpyHostToDevice, s));
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  NCCLCHECK(ncclAllReduce(devSendBuff, devRecvBuff, size, ncclFloat, ncclSum, comm, s));

  CUDACHECK(cudaMemcpyAsync(hostRecvBuff, devRecvBuff, bytes, cudaMemcpyDeviceToHost, s));
  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));
  for (int i = 0; i < size; ++i) {
    printf("%f ", hostRecvBuff[i]);
  }
  printf("\n");
  //free device buffers
  CUDACHECK(cudaFree(devSendBuff));
  CUDACHECK(cudaFree(devRecvBuff));
  //finalizing NCCL
  ncclCommDestroy(comm);
  //finalizing MPI
  MPICHECK(MPI_Finalize());
  printf("[MPI Rank %d] Success \n", myRank);
  return 0;
}
