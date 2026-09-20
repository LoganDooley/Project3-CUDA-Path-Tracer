#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "pathTraceCommon.h"
#include "camera.h"

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
	int activePathCount,
	Geom* dev_geometry,
	int geometryCount
);

__global__ void kernShade(
	PathState* dev_pathStates,
	IntersectionData* dev_intersectionData,
	Material* dev_materials,
	int materialCount,
	int activePathCount,
	cudaSurfaceObject_t surface,
	int width,
	int iteration
);

__global__ void kernColorSurface(cudaSurfaceObject_t surface, PathState* dev_pathStates, int activePathCount, int width);

__global__ void kernDebugRays(PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height);

__global__ void fillSurfaceColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b);

void launchCameraRayGenKernel(PathState* dev_pathStates,
	int width, int height,
	const Camera& camera);

void launchIntersectKernel(PathState* dev_pathStates,
	IntersectionData* dev_intersectionData, Geom* dev_geometry,
	int geometryCount,
	int activePathCount);

void launchShadeKernel(PathState* dev_pathStates,
	IntersectionData* dev_intersectionData,
	Material* dev_materials,
	int materialCount,
	int activePathCount,
	cudaSurfaceObject_t surface,
	int width,
	int iteration);

void launchColorSurfaceKernel(PathState* dev_pathStates,
	int activePathCount,
	cudaSurfaceObject_t surface,
	int width);

void launchDebugRaysKernel(PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height);

void launchColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b);

int runStreamCompaction(PathState* dev_pathStates, int numActivePaths);