#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "pathTraceCommon.h"

__global__ void kernGenerateCameraRays(
	Ray* dev_rays,
	PathState* dev_pathStates,
	int width, int height,
	glm::vec3 cameraPos,
	glm::vec3 cameraLook,
	glm::vec3 cameraRight,
	glm::vec3 cameraUp,
	float fovY);

__global__ void kernIntersect(
	Ray* dev_rays,
	PathState* dev_pathStates,
	int n
);

__global__ void kernColorSurface(cudaSurfaceObject_t surface, Ray* dev_rays, PathState* dev_pathStates, int n, int width);

__global__ void kernDebugRays(Ray* dev_rays, PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height);

__global__ void fillSurfaceColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b);

void launchCameraRayGenKernel(Ray* dev_rays,
	PathState* dev_pathStates,
	int width, int height,
	glm::vec3 cameraPos,
	glm::vec3 cameraLook,
	glm::vec3 cameraRight,
	glm::vec3 cameraUp,
	float fovY);

void launchIntersectKernel(Ray* dev_rays,
	PathState* dev_pathStates,
	int width, int height);

void launchColorSurfaceKernel(Ray* dev_rays,
	PathState* dev_pathStates,
	int width, int height,
	cudaSurfaceObject_t surface);

void launchDebugRaysKernel(Ray* dev_rays, PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height);

void launchColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b);