#pragma once

#include <glm/glm.hpp>

#include "pathTraceCommon.h"
#include "scene.h"

#include "camera.h"
#include "svgfSettings.h"
#include <optional>

// SVGF needs to compute direct and indirect separately, so create a channel struct
// to store all the buffers needed for the reconstruction filter
struct SVGFChannel {
	// Illumination buffers
	glm::vec4* dev_illuminationPrev = nullptr;
	glm::vec4* dev_pingBuffer = nullptr;
	glm::vec4* dev_pongBuffer = nullptr;

	// Variance & Luminance moment buffers
	glm::vec2* dev_moments = nullptr;
	glm::vec2* dev_momentsPrev = nullptr;
	float* dev_variancePing = nullptr;
	float* dev_variancePong = nullptr;
	float* dev_prefilteredVariance = nullptr;

	void allocate(size_t numPixels);
	void free();
	void swapBuffers();
};

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
		const Camera& camera,
		const std::unique_ptr<Scene>& scene);

	void evaluate();

	void debugNormals(cudaSurfaceObject_t surface);

	void debugMotionVectors(cudaSurfaceObject_t surface);

	void debugIlluminance(cudaSurfaceObject_t surface);

	void debugVariance(cudaSurfaceObject_t surface, bool bIndirect = false);

	void drawSettingsImGui();

	// Current G Buffers
	glm::vec4* dev_gBuffer_normalDepth = nullptr;
	glm::vec2* dev_gBuffer_motionVectors = nullptr;
	unsigned int* dev_gBuffer_historyLength = nullptr;
	glm::vec3* dev_gBuffer_albedo = nullptr; // Albedo of first hit so we can demodulate and modulate after filtering

	// Previous G Buffers
	glm::vec4* dev_gBuffer_normalDepthPrev = nullptr;
	unsigned int* dev_gBuffer_historyLengthPrev = nullptr;

	// Direct and indirect lighting are filtered separately
	SVGFChannel directChannel;
	SVGFChannel indirectChannel;

	// Filtered direct + indirect
	glm::vec4* dev_outputColor = nullptr;

private:
	// SVGF Pipeline steps
	void demodulateAlbedo();
	void executeTemporalAccumulation();
	void executeVarianceEstimation();
	void executeAtrousFilteringPipeline();
	void combineChannels();

	void allocateBuffers();
	void freeBuffers();

	int m_width = 0;
	int m_height = 0;

	std::optional<glm::mat4> m_prevViewProj;

	SVGFSettings m_svgfSettings;
};