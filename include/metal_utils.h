#ifndef METAL_UTILS_H
#define METAL_UTILS_H

#include <Metal/Metal.h>

void convertToHalf(const float* src, __fp16* dst, size_t size);
void convertToFloat(const __fp16* src, float* dst, size_t size);

#endif // METAL_UTILS_H