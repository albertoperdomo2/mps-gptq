#import <Metal/Metal.h>
#import "metal_utils.h"

void convertToHalf(const float* src, __fp16* dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = __fp16(src[i]);
    }
}

void convertToFloat(const __fp16* src, float* dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = (float)src[i];
    }
}