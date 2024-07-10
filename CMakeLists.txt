cmake_minimum_required(VERSION 3.5)
project(cuda CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_executable(processing-streams processing-streams.cu)
add_executable(reduce reduce.cu)
