#pragma once

#include <glm/glm.hpp>

#include "pathTraceCommon.h"

#include "camera.h"
#include <optional>

__global__ void kernCaptureGBuffer(
	const PathState* dev_pathStates,
	const IntersectionData* dev_intersectionData,
	glm::vec4* dev_gBuffer_normalDepth,
	glm::vec2* dev_gBuffer_motionVectors,
	glm::vec3* dev_worldPositions, 
	glm::mat4 currentViewProj,
	glm::mat4 prevViewProj,
	int width,
	int height,
	int currentActivePathCount);

__global__ void kernTemporalAccumulation(
	const glm::vec4* dev_gBuffer_normalDepth,
	const glm::vec2* dev_gBuffer_motionVectors,
	const glm::vec4* dev_gBuffer_normalDepthPrev,
	const glm::vec4* dev_pingBuffer,
	const glm::vec4* dev_illuminationPrev,
	const glm::vec2* dev_momentsPrev,
	glm::vec4* dev_integratedColor,
	glm::vec2* dev_moments,
	unsigned int* dev_historyLength,
	const unsigned int* dev_historyLengthPrev,
	int width, int height);

class SVGFManager {
public:
	SVGFManager();
	SVGFManager(int width, int height);
	~SVGFManager();

	SVGFManager& operator=(const SVGFManager&) = delete;
	SVGFManager(const SVGFManager&) = delete;

	SVGFManager& operator=(SVGFManager&&) noexcept;
	SVGFManager(SVGFManager&&) noexcept;

	void swapBuffers();

	void resize(int width, int height);

	void captureGBuffer(
		const PathState* dev_pathStates,
		const IntersectionData* dev_intersectionData,
		int activePathCount,
		const Camera& camera);

	void executeTemporalAccumulation();

	void executeVarianceEstimation();

	void executeAtrousFilteringPipeline();

	void debugNormals(cudaSurfaceObject_t surface);

	void debugMotionVectors(cudaSurfaceObject_t surface);

	void debugIlluminance(cudaSurfaceObject_t surface);

	void debugVariance(cudaSurfaceObject_t surface);

	// Current G Buffers
	glm::vec4* dev_gBuffer_normalDepth = nullptr;
	glm::vec2* dev_gBuffer_motionVectors = nullptr;
	unsigned int* dev_gBuffer_historyLength = nullptr;

	// Previous G Buffers
	glm::vec4* dev_gBuffer_normalDepthPrev = nullptr;
	unsigned int* dev_gBuffer_historyLengthPrev = nullptr;

	// Illumination buffers
	glm::vec4* dev_illuminationPrev = nullptr;
	glm::vec4* dev_pingBuffer = nullptr;
	glm::vec4* dev_pongBuffer = nullptr;

	// Variance & Luminance moments
	glm::vec2* dev_moments = nullptr;
	glm::vec2* dev_momentsPrev = nullptr;
	float* dev_variancePing = nullptr;
	float* dev_variancePong = nullptr;
	float* dev_prefilteredVariance = nullptr;

private:
	void allocateBuffers();
	void freeBuffers();

	int m_width = 0;
	int m_height = 0;

	std::optional<glm::mat4> m_prevViewProj;
};