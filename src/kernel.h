#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "pathTraceCommon.h"

__global__ void kernGenerateCameraRays(
	PathState* dev_pathStates,
	int width, int height,
	glm::vec3 cameraPos,
	glm::vec3 cameraLook,
	glm::vec3 cameraRight,
	glm::vec3 cameraUp,
	float fovY);

__global__ void kernIntersect(
	PathState* dev_pathStates,
	IntersectionData* dev_intersectionData,
	int n
);

__global__ void kernColorSurface(cudaSurfaceObject_t surface, PathState* dev_pathStates, int n, int width);

__global__ void kernDebugRays(PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height);

__global__ void fillSurfaceColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b);

void launchCameraRayGenKernel(PathState* dev_pathStates,
	int width, int height,
	glm::vec3 cameraPos,
	glm::vec3 cameraLook,
	glm::vec3 cameraRight,
	glm::vec3 cameraUp,
	float fovY);

void launchIntersectKernel(PathState* dev_pathStates,
	IntersectionData* dev_intersectionData,
	int width, int height);

void launchColorSurfaceKernel(PathState* dev_pathStates,
	int width, int height,
	cudaSurfaceObject_t surface);

void launchDebugRaysKernel(PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height);

void launchColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b);