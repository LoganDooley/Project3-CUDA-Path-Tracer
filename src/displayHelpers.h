#pragma once

#include <cuda_runtime.h>

#include <glm/glm.hpp>

// Writes a [0, 1] color to the surface in BGRA8 format
__device__ inline void writeSurfacePixel(cudaSurfaceObject_t surface, int x, int y, const glm::vec3& color) {
	glm::vec3 clamped = glm::clamp(color, 0.0f, 1.0f);

	uchar4 pixelColor;
	pixelColor.x = (unsigned char)(clamped.b * 255.0f);
	pixelColor.y = (unsigned char)(clamped.g * 255.0f);
	pixelColor.z = (unsigned char)(clamped.r * 255.0f);
	pixelColor.w = 255;

	surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
}
