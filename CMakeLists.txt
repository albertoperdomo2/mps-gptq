cmake_minimum_required(VERSION 3.10)
project(MPS-GPTQ)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_library(METAL Metal)
find_library(METALKIT MetalKit)
find_library(FOUNDATION Foundation)

# NOTE: New shader files are added here
add_executable(metal_gptq src/main.mm src/metal_gemm.mm src/metal_add.mm src/metal_hessian.mm src/metal_quantize_layer.mm src/metal_utils.mm)

target_include_directories(metal_gptq PRIVATE include)
target_link_libraries(metal_gptq ${METAL} ${METALKIT} ${FOUNDATION})

configure_file(${CMAKE_SOURCE_DIR}/src/metal_add.metal ${CMAKE_BINARY_DIR}/src/metal_add.metal COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/src/metal_gemm.metal ${CMAKE_BINARY_DIR}/src/metal_gemm.metal COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/src/metal_hessian.metal ${CMAKE_BINARY_DIR}/src/metal_hessian.metal COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/src/metal_quantize_layer.metal ${CMAKE_BINARY_DIR}/src/metal_quantize_layer.metal COPYONLY)