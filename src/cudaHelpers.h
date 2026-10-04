#pragma once

#include <cuda_runtime.h>

// Integer division rounded up, for sizing grids that cover every element
__host__ __device__ inline int divup(int a, int b) {
	return (a + b - 1) / b;
}

// Grid with enough blocks to cover a width x height image
inline dim3 make2DGrid(int width, int height, const dim3& blockSize) {
	return dim3(divup(width, blockSize.x), divup(height, blockSize.y));
}
