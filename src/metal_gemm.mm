#import <Metal/Metal.h>
#include "metal_gemm.h"
#import "metal_utils.h"

#define TILE_SIZE 32

void gemm_metal(float* A, float* B, float* C, int M, int N, int K) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (!device) {
        printf("Metal is not supported on this device.\n");
        return;
    }

    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    NSError *error = nil;
    NSString *shaderSource = [NSString stringWithContentsOfFile:@"src/metal_gemm.metal" encoding:NSUTF8StringEncoding error:&error];

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];

    if (!library) {
        printf("Failed to load metal library: %s\n", error.localizedDescription.UTF8String);
        return;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"gemm"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

    __fp16* hA = new __fp16[M * K];
    __fp16* hB = new __fp16[K * N];

    // convert to half
    convertToHalf(A, hA, M * K);
    convertToHalf(B, hB, K * N);

    id<MTLBuffer> bufferA = [device newBufferWithBytes:hA length:M * K * sizeof(__fp16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithBytes:hB length:K * N * sizeof(__fp16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:M * N * sizeof(__fp16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferK = [device newBufferWithBytes:&K length:sizeof(uint) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferN = [device newBufferWithBytes:&N length:sizeof(uint) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferM = [device newBufferWithBytes:&M length:sizeof(uint) options:MTLResourceStorageModeShared];

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferC offset:0 atIndex:2];
    [encoder setBuffer:bufferK offset:0 atIndex:3];
    [encoder setBuffer:bufferN offset:0 atIndex:4];
    [encoder setBuffer:bufferM offset:0 atIndex:5];

    MTLSize gridSize = MTLSizeMake(M, N, 1);
    MTLSize threadGroupSize = MTLSizeMake(TILE_SIZE, TILE_SIZE, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // convert back
    __fp16* halfResults = (__fp16*)bufferC.contents;
    convertToFloat(halfResults, C, M * N);

    // memcpy(C, bufferC.contents, M * N * sizeof(__fp16));

    delete[] hA;
    delete[] hB;
}
