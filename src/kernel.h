#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void fillSurfaceColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b);

void launchColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b);