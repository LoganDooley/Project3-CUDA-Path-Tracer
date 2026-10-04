#pragma once

#include <cuda_runtime.h>

__host__ __device__ inline int divup(int a, int b) {
	return (a + b - 1) / b;
}

// Create enough blocks to cover a 2D grid of the desired size
inline dim3 make2DGrid(int width, int height, const dim3& blockSize) {
	return dim3(divup(width, blockSize.x), divup(height, blockSize.y));
}
