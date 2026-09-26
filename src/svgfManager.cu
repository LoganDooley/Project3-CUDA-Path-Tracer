#include "svgfManager.h"

#include <cuda_runtime.h>

#include <iostream>

#define CUDA_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << " -> Line: " << line << std::endl;
		if (abort) exit(code);
	}
}

__device__ glm::vec2 calculateMotionVector(
	glm::vec3 worldPos,
	glm::mat4 currentViewProj,
	glm::mat4 prevViewProj,
	int width,
	int height) 
{
	glm::vec4 currentClip = currentViewProj * glm::vec4(worldPos, 1.0f);
	glm::vec4 prevClip = prevViewProj * glm::vec4(worldPos, 1.0f);

	if (currentClip.w == 0.0f || prevClip.w == 0.0f) {
		return glm::vec2(0.0f);
	}

	glm::vec2 currentNDC = glm::vec2(currentClip.x / currentClip.w, currentClip.y / currentClip.w);
	glm::vec2 prevNDC = glm::vec2(prevClip.x / prevClip.w, prevClip.y / prevClip.w);

	glm::vec2 currentScreen = glm::vec2(
		(currentNDC.x + 1.0f) * 0.5f * width,
		(1.0f - currentNDC.y) * 0.5f * height
	);

	glm::vec2 prevScreen = glm::vec2(
		(prevNDC.x + 1.0f) * 0.5f * width,
		(1.0f - prevNDC.y) * 0.5f * height
	);

	return prevScreen - currentScreen;
}

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
	int currentActivePathCount)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= currentActivePathCount) {
		return;
	}

	const PathState& pathState = dev_pathStates[index];
	const IntersectionData& intersectionData = dev_intersectionData[index];

	int pixelIdx = pathState.pixelIndex;

	glm::vec3 normal = intersectionData.normal;
	glm::vec3 worldPos = pathState.ray.getPositionAtTime(intersectionData.t);
	float depth = intersectionData.t;

	if (depth < 0.0f || depth >= 1e10f) {
		dev_gBuffer_normalDepth[pixelIdx] = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);
		dev_gBuffer_motionVectors[pixelIdx] = glm::vec2(0.0f);
		return;
	}

	dev_gBuffer_normalDepth[pixelIdx] = glm::vec4(normal, depth);
	
	glm::vec2 motionVector = calculateMotionVector(worldPos, currentViewProj, prevViewProj, width, height);
	dev_gBuffer_motionVectors[pixelIdx] = motionVector;

	if (dev_worldPositions != nullptr) {
		dev_worldPositions[pixelIdx] = worldPos;
	}
}

__device__ void drawVec3ToSurface(const glm::vec3& vector, cudaSurfaceObject_t surface, int x, int y) {
	float r = vector.x * 0.5f + 0.5f;
	float g = vector.y * 0.5f + 0.5f;
	float b = vector.z * 0.5f + 0.5f;

	uchar4 pixelColor;
	pixelColor.x = (unsigned char)(b * 255.0f); // Blue
	pixelColor.y = (unsigned char)(g * 255.0f); // Green
	pixelColor.z = (unsigned char)(r * 255.0f); // Red
	pixelColor.w = 255;

	surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
}

__global__ void kernDebugNormals(glm::vec4* dev_gBuffer_normalDepth, cudaSurfaceObject_t surface, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = y * width + x;
	const glm::vec4& normalDepth = dev_gBuffer_normalDepth[pixelIndex];

	drawVec3ToSurface(normalDepth, surface, x, y);
}

__global__ void kernDebugMotionVectors(glm::vec2* dev_gBuffer_motionVectors, cudaSurfaceObject_t surface, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = y * width + x;
	const glm::vec2& motionVector = dev_gBuffer_motionVectors[pixelIndex];

	glm::vec3 visualVector = glm::vec3(motionVector.x * 0.05f, motionVector.y * 0.05f, 0.0f);
	drawVec3ToSurface(visualVector, surface, x, y);
}

SVGFManager::SVGFManager()
{
}

SVGFManager::SVGFManager(int width, int height) :
	SVGFManager()
{
	resize(width, height);
}

SVGFManager::~SVGFManager()
{
	freeBuffers();
}

SVGFManager& SVGFManager::operator=(SVGFManager&& other) noexcept
{
	if (this != &other) {
		freeBuffers();

		m_width = other.m_width;
		m_height = other.m_height;

		dev_gBuffer_normalDepth = other.dev_gBuffer_normalDepth;
		dev_gBuffer_motionVectors = other.dev_gBuffer_motionVectors;
		dev_gBuffer_historyLength = other.dev_gBuffer_historyLength;
		dev_gBuffer_normalDepthPrev = other.dev_gBuffer_normalDepthPrev;
		dev_illuminationPrev = other.dev_illuminationPrev;
		dev_pingBuffer = other.dev_pingBuffer;
		dev_pongBuffer = other.dev_pongBuffer;
		dev_moments = other.dev_moments;
		dev_momentsPrev = other.dev_momentsPrev;
		dev_variance = other.dev_variance;

		other.m_width = 0;
		other.m_height = 0;
		other.dev_gBuffer_normalDepth = nullptr;
		other.dev_gBuffer_motionVectors = nullptr;
		other.dev_gBuffer_historyLength = nullptr;
		other.dev_gBuffer_normalDepthPrev = nullptr;
		other.dev_illuminationPrev = nullptr;
		other.dev_pingBuffer = nullptr;
		other.dev_pongBuffer = nullptr;
		other.dev_moments = nullptr;
		other.dev_momentsPrev = nullptr;
		other.dev_variance = nullptr;
	}

	return *this;
}

SVGFManager::SVGFManager(SVGFManager&& other) noexcept
{
	*this = std::move(other);
}

void SVGFManager::swapBuffers()
{
	std::swap(dev_gBuffer_normalDepth, dev_gBuffer_normalDepthPrev);
	std::swap(dev_moments, dev_momentsPrev);
	std::swap(dev_illuminationPrev, dev_pongBuffer);
}

void SVGFManager::resize(int width, int height)
{
	if (m_width == width && m_height == height) {
		return;
	}

	freeBuffers();
	
	m_width = width;
	m_height = height;

	if (m_width > 0 && m_height > 0) {
		allocateBuffers();
	}
}

void SVGFManager::captureGBuffer(
	const PathState* dev_pathStates,
	const IntersectionData* dev_intersectionData,
	int activePathCount,
	const Camera& camera)
{
	if (activePathCount <= 0) return;

	int blockSize = 256;
	int gridSize = (activePathCount + blockSize - 1) / blockSize;

	glm::mat4 currentViewProj = camera.getViewProjectionMatrix(m_width, m_height);
	glm::mat4 prevViewProj = m_prevViewProj.has_value() ? m_prevViewProj.value() : currentViewProj;

	kernCaptureGBuffer << <gridSize, blockSize >> > (
		dev_pathStates,
		dev_intersectionData,
		dev_gBuffer_normalDepth,
		dev_gBuffer_motionVectors,
		nullptr,
		currentViewProj,
		prevViewProj,
		m_width,
		m_height,
		activePathCount
		);

	m_prevViewProj = currentViewProj;
}

void SVGFManager::debugNormals(cudaSurfaceObject_t surface)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);

	kernDebugNormals << <gridSize, blockSize >> > (dev_gBuffer_normalDepth, surface, m_width, m_height);
}

void SVGFManager::debugMotionVectors(cudaSurfaceObject_t surface)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);

	kernDebugMotionVectors << <gridSize, blockSize >> > (dev_gBuffer_motionVectors, surface, m_width, m_height);
}

void SVGFManager::allocateBuffers()
{
	size_t numPixels = static_cast<size_t>(m_width) * m_height;

	CUDA_CHECK(cudaMalloc(&dev_gBuffer_normalDepth, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_normalDepthPrev, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_motionVectors, numPixels * sizeof(glm::vec2)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_historyLength, numPixels * sizeof(unsigned int)));

	CUDA_CHECK(cudaMalloc(&dev_illuminationPrev, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_pingBuffer, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_pongBuffer, numPixels * sizeof(glm::vec4)));

	CUDA_CHECK(cudaMalloc(&dev_moments, numPixels * sizeof(glm::vec2)));
	CUDA_CHECK(cudaMalloc(&dev_momentsPrev, numPixels * sizeof(glm::vec2)));
	CUDA_CHECK(cudaMalloc(&dev_variance, numPixels * sizeof(float)));

	// Initialize history tracker sizes to 0
	CUDA_CHECK(cudaMemset(dev_gBuffer_historyLength, 0, numPixels * sizeof(unsigned int)));
}

void SVGFManager::freeBuffers()
{
	if (dev_gBuffer_normalDepth) {
		cudaFree(dev_gBuffer_normalDepth);
	}

	if (dev_gBuffer_normalDepthPrev) {
		cudaFree(dev_gBuffer_normalDepthPrev);
	}

	if (dev_gBuffer_motionVectors) {
		cudaFree(dev_gBuffer_motionVectors);
	}

	if (dev_gBuffer_historyLength) {
		cudaFree(dev_gBuffer_historyLength);
	}

	if (dev_illuminationPrev) {
		cudaFree(dev_illuminationPrev);
	}

	if (dev_pingBuffer) {
		cudaFree(dev_pingBuffer);
	}

	if (dev_pongBuffer) {
		cudaFree(dev_pongBuffer);
	}

	if (dev_moments) {
		cudaFree(dev_moments);
	}

	if (dev_momentsPrev) {
		cudaFree(dev_momentsPrev);
	}

	if (dev_variance) {
		cudaFree(dev_variance);
	}

	dev_gBuffer_normalDepth = nullptr; 
	dev_gBuffer_normalDepthPrev = nullptr;
	dev_gBuffer_motionVectors = nullptr; 
	dev_gBuffer_historyLength = nullptr;
	dev_illuminationPrev = nullptr;
	dev_pingBuffer = nullptr; 
	dev_pongBuffer = nullptr;
	dev_moments = nullptr; 
	dev_momentsPrev = nullptr;
	dev_variance = nullptr;
}