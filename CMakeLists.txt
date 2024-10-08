cmake_minimum_required(VERSION 3.5)
project(cuda CXX C CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CUTLASS_INCLUDE ${CUTLASS}/include)
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr)
message("cutlass include: ${CUTLASS_INCLUDE}")
include_directories(${CUTLASS_INCLUDE})


add_executable(processing-streams processing-streams.cu)
add_executable(reduce reduce.cu)
add_executable(add-inline-asm add-inline-asm.cu)
add_executable(wmma wmma.cu)
add_executable(ldmatrix ldmatrix.cu)
add_executable(mma mma.cu)
add_executable(device-query device-query.cu)

add_subdirectory(load-store-vectorization)
add_subdirectory(sumMatrix)
add_subdirectory(transpose)
add_subdirectory(share-memory)
add_subdirectory(cutlass)
