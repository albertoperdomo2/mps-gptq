cmake_minimum_required(VERSION 3.10)
project(MPS-GPTQ)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_library(METAL Metal)
find_library(METALKIT MetalKit)
find_library(FOUNDATION Foundation)

# NOTE: New shader files are added here
add_executable(metal_gptq src/main.mm src/metal_gemm.mm src/metal_add.mm)

target_include_directories(metal_gptq PRIVATE include)
target_link_libraries(metal_gptq ${METAL} ${METALKIT} ${FOUNDATION})

configure_file(${CMAKE_SOURCE_DIR}/src/metal_add.metal ${CMAKE_BINARY_DIR}/src/metal_add.metal COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/src/metal_gemm.metal ${CMAKE_BINARY_DIR}/src/metal_gemm.metal COPYONLY)

# cmake_minimum_required(VERSION 3.14)
# project(MPS-GPTQ)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# # Enable Objective-C++
# enable_language(OBJCXX)

# # Add source files
# add_executable(metal_gptq src/main.mm src/metal_add.mm)
# target_include_directories(metal_gptq PRIVATE include)
# target_link_libraries(metal_gptq "-framework Metal" "-framework Foundation")

# # CMake does not automatically copy non-source files into the build directory, we must specify it manually
# configure_file(${CMAKE_SOURCE_DIR}/src/metal_add.metal ${CMAKE_BINARY_DIR}/src/metal_add.metal COPYONLY)