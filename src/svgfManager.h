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

	void debugNormals(cudaSurfaceObject_t surface);

	void debugMotionVectors(cudaSurfaceObject_t surface);

	// Current G Buffers
	glm::vec4* dev_gBuffer_normalDepth = nullptr;
	glm::vec2* dev_gBuffer_motionVectors = nullptr;
	unsigned int* dev_gBuffer_historyLength = nullptr;

	// Previous G Buffers
	glm::vec4* dev_gBuffer_normalDepthPrev = nullptr;

	// Illumination buffers
	glm::vec4* dev_illuminationPrev = nullptr;
	glm::vec4* dev_pingBuffer = nullptr;
	glm::vec4* dev_pongBuffer = nullptr;

	// Variance & Luminance moments
	glm::vec2* dev_moments = nullptr;
	glm::vec2* dev_momentsPrev = nullptr;
	float* dev_variance = nullptr;

private:
	void allocateBuffers();
	void freeBuffers();

	int m_width = 0;
	int m_height = 0;

	std::optional<glm::mat4> m_prevViewProj;
};