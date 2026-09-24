#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "pathTraceCommon.h"
#include "camera.h"
#include "scene.h"
#include "environmentMap.h"

#include <memory>

__global__ void kernGenerateCameraRays(
	PathState* dev_pathStates,
	int width, int height,
	glm::vec3 cameraPos,
	glm::vec3 cameraLook,
	glm::vec3 cameraRight,
	glm::vec3 cameraUp,
	float fovY,
	int frameIndex);

__global__ void kernIntersect(
	PathState* dev_pathStates,
	IntersectionData* dev_intersectionData,
	int activePathCount,
	DevScene dev_scene
);

__global__ void kernShade(
	PathState* dev_pathStates,
	IntersectionData* dev_intersectionData,
	DevScene dev_scene,
	cudaTextureObject_t environmentMap,
	int activePathCount,
	cudaSurfaceObject_t surface,
	glm::vec3* dev_accumulatedColor,
	unsigned int* dev_sampleCounts,
	int width,
	int iteration,
	int frameIndex
);

__global__ void kernColorSurface(
	cudaSurfaceObject_t surface, 
	glm::vec3* dev_accumulatedColor,
	unsigned int* dev_sampleCounts, 
	PathState* dev_pathStates, 
	int activePathCount, 
	int width);

__global__ void kernDebugRays(PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height);

__global__ void fillSurfaceColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b);

void launchCameraRayGenKernel(PathState* dev_pathStates,
	int width, int height,
	const Camera& camera,
	int frameIndex);

void launchIntersectKernel(PathState* dev_pathStates,
	IntersectionData* dev_intersectionData, 
	const std::unique_ptr<Scene>& scene,
	int activePathCount);

void launchShadeKernel(PathState* dev_pathStates,
	IntersectionData* dev_intersectionData,
	const std::unique_ptr<Scene>& scene,
	const std::unique_ptr<EnvironmentMap>& environmentMap,
	int activePathCount,
	cudaSurfaceObject_t surface,
	glm::vec3* dev_accumulatedColor,
	unsigned int* dev_sampleCounts,
	int width,
	int iteration,
	int frameIndex);

void launchColorSurfaceKernel(PathState* dev_pathStates,
	int activePathCount,
	cudaSurfaceObject_t surface,
	glm::vec3* dev_accumulatedColor,
	unsigned int* dev_sampleCounts,
	int width);

void launchDebugRaysKernel(PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height);

void launchColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b);

int runStreamCompaction(PathState* dev_pathStates, IntersectionData* dev_intersectionData, int numActivePaths);