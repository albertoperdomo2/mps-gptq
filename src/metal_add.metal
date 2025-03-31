#include <metal_stdlib>
using namespace metal;

// Uses thread_position_in_grid: each thread computes one element

kernel void add_arrays(
  device const float* inA,
  device const float* inB,
  device float* result,
  uint id [[thread_position_in_grid]]
  ) {
    result[id] = inA[id] + inB[id];
}
