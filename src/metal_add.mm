#import <Metal/Metal.h>
#include "metal_add.h"

void add_arrays_metal(float* inA, float* inB, float* out, int size) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (!device) {
        printf("Metal is not supported on this device.\n");
        return;
    }

    // Create command queue
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];

    // Load Metal lib
    NSError *error = nil;
    NSString *shaderSource = [NSString stringWithContentsOfFile:@"src/metal_add.metal" encoding:NSUTF8StringEncoding error:&error];

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];

    if (!library) {
        printf("Failed to load metal library: %s\n", error.localizedDescription.UTF8String);
        return;
    }

    id<MTLFunction> function = [library newFunctionWithName:@"add_arrays"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];

    // Create buffers
    id<MTLBuffer> bufferA = [device newBufferWithBytes:inA length:size * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithBytes:inB length:size * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferOut = [device newBufferWithLength:size * sizeof(float) options:MTLResourceStorageModeShared];

    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferOut offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(size, 1, 1);
    MTLSize threadGroupSize = MTLSizeMake(1, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    memcpy(out, bufferOut.contents, size * sizeof(float));
}
