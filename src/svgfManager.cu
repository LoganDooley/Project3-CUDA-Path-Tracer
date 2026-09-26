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

__device__ float computeLuminance(const glm::vec3& color)
{
	glm::vec3 luminanceWeights = glm::vec3(0.2126f, 0.7152f, 0.0722f);
	float luminance = glm::dot(color, luminanceWeights);
	return luminance;
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

__global__ void kernTemporalAccumulation(
	const glm::vec4* dev_gBuffer_normalDepth,
	const glm::vec2* dev_gBuffer_motionVectors,
	const glm::vec4* dev_gBuffer_normalDepthPrev,
	const glm::vec4* dev_pingBuffer,      
	const glm::vec4* dev_illuminationPrev,
	const glm::vec2* dev_momentsPrev,

	// Output Buffers
	glm::vec4* dev_integratedColor,
	glm::vec2* dev_moments,
	unsigned int* dev_historyLength,
	const unsigned int* dev_historyLengthPrev,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = y * width + x;

	glm::vec4 currentND = dev_gBuffer_normalDepth[pixelIndex];
	float currentDepth = currentND.w;

	if (currentDepth < 0.0f) {
		// TODO: This won't work with environment maps
		dev_integratedColor[pixelIndex] = dev_pingBuffer[pixelIndex];
		dev_moments[pixelIndex] = glm::vec2(0.0f);
		dev_historyLength[pixelIndex] = 0;
		return;
	}

	glm::vec3 currentNormal = glm::vec3(currentND.x, currentND.y, currentND.z);
	glm::vec4 currentColor = dev_pingBuffer[pixelIndex];

	float currentLuminance = computeLuminance(currentColor);
	currentLuminance = glm::clamp(currentLuminance, 0.0f, 30.f);

	// Get prev x and y from motion vector
	glm::vec2 motionVector = dev_gBuffer_motionVectors[pixelIndex];
	int prevX = __float2int_rn((float)x + motionVector.x);
	int prevY = __float2int_rn((float)y + motionVector.y);

	bool bValidHistory = false;
	float alpha = 1.0f;
	unsigned int updatedHistoryCount = 1;
	int prevPixelIndex = -1;

	if (prevX >= 0 && prevX < width && prevY >= 0 && prevY < height) {
		// Prev is samplable

		prevPixelIndex = prevY * width + prevX;

		glm::vec4 prevND = dev_gBuffer_normalDepthPrev[prevPixelIndex];
		float prevDepth = prevND.w;
		glm::vec3 prevNormal = glm::vec3(prevND.x, prevND.y, prevND.z);

		float normalDot = glm::dot(currentNormal, prevNormal);

		float depthDiff = glm::abs(currentDepth - prevDepth) / glm::max(currentDepth, 1e-4f);

		// We want the normals to be very similar and the difference in depth to be somewhat small
		if (normalDot > 0.95f && depthDiff < 0.1f && prevDepth >= 0.0f) {
			bValidHistory = true;
			unsigned int historicalCount = dev_historyLengthPrev[prevPixelIndex];

			updatedHistoryCount = glm::min(historicalCount + 1, 32U);

			alpha = glm::max(1.0f / (float)updatedHistoryCount, 0.05f);
		}
	}

	dev_historyLength[pixelIndex] = updatedHistoryCount;

	glm::vec2 currentMoments = glm::vec2(currentLuminance, currentLuminance * currentLuminance);

	// Write integrated color and moments back to buffers
	if (bValidHistory && prevPixelIndex >= 0) {
		dev_integratedColor[pixelIndex] = glm::mix(dev_illuminationPrev[prevPixelIndex], currentColor, alpha);

		dev_moments[pixelIndex] = glm::mix(dev_momentsPrev[prevPixelIndex], currentMoments, alpha);
	}
	else {
		dev_integratedColor[pixelIndex] = currentColor;

		dev_moments[pixelIndex] = currentMoments;
	}
}

__device__ float calculateVariance(const glm::vec2& moments) {
	return glm::max(0.0f, moments.x - moments.y * moments.y);
}

__global__ void kernEstimateVariance(
	const glm::vec4* dev_pingBuffer,
	const glm::vec2* dev_moments,
	const glm::vec4* dev_gBuffer_normalDepth,
	const unsigned int* dev_historyLength,
	float* dev_variance,
	int width, int height
)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int pixelIndex = y * width + x;
	unsigned int history = dev_historyLength[pixelIndex];

	// Need some significant history to calculate variance
	if (history >= 4) {
		glm::vec2 m = dev_moments[pixelIndex];

		dev_variance[pixelIndex] = calculateVariance(m);
		return;
	}
	dev_variance[pixelIndex] = 0.0f;
	return;

	// If short history, use spatial varaince
	glm::vec4 centerND = dev_gBuffer_normalDepth[pixelIndex];
	float centerDepth = centerND.w;

	if (centerDepth < 0.0f) {
		dev_variance[pixelIndex] = 0.0f;
		return;
	}

	glm::vec3 centerNormal = glm::vec3(centerND);

	glm::vec2 sumMoments = glm::vec2(0.0f);
	float sumWeight = 0.0f;

	// 7x7 bilateral filter
	for (int dy = -3; dy <= 3; dy++) {
		for (int dx = -3; dx <= 3; dx++) {
			int nx = x + dx;
			int ny = y + dy;

			if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
				continue;
			}

			int neighborIdx = ny * width + nx;
			glm::vec4 neighborND = dev_gBuffer_normalDepth[neighborIdx];
			float neighborDepth = neighborND.w;

			if (neighborDepth < 0.0f) {
				continue;
			}

			glm::vec3 neighborNormal = glm::vec3(neighborND);

			// Bilateral weights
			float wNormal = glm::pow(glm::max(0.0f, glm::dot(centerNormal, neighborNormal)), 128.0f);
			float wDepth = glm::exp(-glm::abs(centerDepth - neighborDepth) / (centerDepth * 0.1f + 1e-4f));
			
			unsigned int neighborHistory = dev_historyLength[neighborIdx];
			float wHistory = glm::max(1.0f, (float)neighborHistory);

			float wBilateral = wNormal * wDepth * wHistory;

			glm::vec2 nMoments = dev_moments[neighborIdx];
			sumMoments += wBilateral * nMoments;
			sumWeight += wBilateral;
		}
	}

	if (sumWeight > 0.0f) {
		sumMoments /= sumWeight;
	}

	dev_variance[pixelIndex] = calculateVariance(sumMoments);
}

__global__ void kernVariancePrefilter3x3(
	const float* dev_variance,
	const glm::vec4* dev_gBuffer_normalDepth,
	float* dev_prefilteredVariance,
	int width, int height
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	float centerDepth = dev_gBuffer_normalDepth[pixelIndex].w;
	if (centerDepth < 0.0f) {
		// Skip computing variance
		dev_prefilteredVariance[pixelIndex] = 0.0f;
		return;
	}

	// 3x3 Gaussian kernel
	const float wGaus[3][3] = {
		{ 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f },
		{ 2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f },
		{ 1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f }
	};

	float sumVariance = 0.0f;
	float sumWeight = 0.0f;

	// Blur over 3x3 neighborhood
	for (int r = -1; r <= 1; ++r) {
		for (int c = -1; c <= 1; ++c) {
			int nx = glm::clamp(x + c, 0, width - 1);
			int ny = glm::clamp(y + r, 0, height - 1);
			int neighborIndex = ny * width + nx;

			float neighborDepth = dev_gBuffer_normalDepth[neighborIndex].w;
			if (neighborDepth < 0.0f) {
				// Neighbor missed geometry, skip
				continue;
			}

			float w = wGaus[r + 1][c + 1];
			sumVariance += dev_variance[neighborIndex] * w;
			sumWeight += w;
		}
	}

	if (sumWeight > 0.0f) {
		dev_prefilteredVariance[pixelIndex] = sumVariance / sumWeight;
	}
	else {
		// Copy non blurred variance for rays missing geometry
		dev_prefilteredVariance[pixelIndex] = dev_variance[pixelIndex];
	}
}

__device__ glm::vec2 computeDepthGradient(
	glm::ivec2 p,
	const glm::vec4* dev_gBuffer_normalDepth,
	float depthP,
	int width,
	int height
) {
	int rightX = glm::min(p.x + 1, width - 1);
	int leftX = glm::max(p.x - 1, 0);
	int bottomY = glm::min(p.y + 1, height - 1);
	int topY = glm::max(p.y - 1, 0);

	float depthRight = dev_gBuffer_normalDepth[p.y * width + rightX].w;
	float depthLeft = dev_gBuffer_normalDepth[p.y * width + leftX].w;
	float depthBottom = dev_gBuffer_normalDepth[bottomY * width + p.x].w;
	float depthTop = dev_gBuffer_normalDepth[topY * width + p.x].w;

	float dDepthdX = 0.0f;
	if (depthRight >= 0.0f) {
		dDepthdX = depthRight - depthP;
	}
	else if (depthLeft >= 0.0f) {
		dDepthdX = depthP - depthLeft;
	}

	float dDepthdY = 0.0f;
	if (depthBottom >= 0.0f) {
		dDepthdY = depthBottom - depthP;
	}
	else if (depthTop >= 0.0f) {
		dDepthdY = depthP - depthTop;
	}

	return glm::vec2(dDepthdX, dDepthdY);
}

__device__ float computeTotalWeight(
	// Pixel values
	glm::ivec2 p,
	glm::ivec2 q,
	// Depth values
	float depthP,
	float depthQ,
	float sigmaDepth,
	glm::vec2 depthGradP,
	// Normal values
	glm::vec3 normalP,
	glm::vec3 normalQ,
	float sigmaNormal,
	// Luminance values
	float luminanceP,
	float luminanceQ,
	float luminancePrefilteredVariance,
	float sigmaLuminance
)
{
	const float epsilon = 0.0001f;
	// Calculate w_z
	float depthDelta = glm::abs(depthP - depthQ);
	float w_z = glm::exp(-depthDelta / (sigmaDepth * glm::abs(glm::dot(depthGradP, glm::vec2(p - q))) + epsilon));

	// Calculate w_n
	float normalDot = glm::dot(normalP, normalQ);
	float w_n = glm::pow(glm::max(0.0f, normalDot), sigmaNormal);

	// Calculate w_l
	float luminanceDelta = glm::abs(luminanceP - luminanceQ);
	float w_l = glm::exp(-luminanceDelta / (sigmaLuminance * glm::sqrt(luminancePrefilteredVariance) + epsilon));

	float w_i = w_z * w_n * w_l;

	return w_i;
}

__global__ void kernAtrousFilter(
	const glm::vec4* dev_inputColor,
	const glm::vec4* dev_gBuffer_normalDepth,
	const float* dev_inputVariance,
	const float* dev_prefilteredVariance,
	float* dev_outputVariance,
	glm::vec4* dev_outputColor,
	int stride,
	float sigmaLuminance,
	float sigmaNormal,
	float sigmaDepth,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;
	glm::ivec2 p = glm::ivec2(x, y);

	glm::vec4 centerND = dev_gBuffer_normalDepth[pixelIndex];
	float depthP = centerND.w;

	if (depthP < 0.0f) {
		// Ray missed, just copy without filtering
		dev_outputColor[pixelIndex] = dev_inputColor[pixelIndex];
		return;
	}

	glm::vec2 depthGradP = computeDepthGradient(p, dev_gBuffer_normalDepth, depthP, width, height);
	glm::vec3 normalP = glm::vec3(centerND);
	glm::vec4 colorP = dev_inputColor[pixelIndex];
	float luminanceP = computeLuminance(glm::vec3(colorP));
	float varianceP = dev_inputVariance[pixelIndex];
	float prefilteredVarianceP = dev_prefilteredVariance[pixelIndex];

	// h filter kernel 
	const float h[5] = { 1.0f/16.0f, 1.0f/4.0f, 3.0f/8.0f, 1.0f/4.0f, 1.0f/16.0f };

	glm::vec3 sumColor = glm::vec3(0.0f);
	float sumColorWeight = 0.0f;

	float sumVariance = 0.0f;
	float sumVarianceWeight = 0.0f;

	// Filter over 5x5 window
	for (int r = -2; r <= 2; ++r) {
		for (int c = -2; c <= 2; ++c) {
			if (r == 0 && c == 0) {
				float wKernel = h[2] * h[2];
				sumColor += glm::vec3(colorP) * wKernel;
				sumColorWeight += wKernel;
				sumVariance += wKernel * wKernel * varianceP;
				sumVarianceWeight += wKernel;
				continue;
			}

			// Calculate neighbor index
			int nx = glm::clamp(x + c * stride, 0, width - 1);
			int ny = glm::clamp(y + r * stride, 0, height - 1);
			glm::ivec2 q = glm::ivec2(nx, ny);
			int neighborIndex = ny * width + nx;



			glm::vec4 neighborND = dev_gBuffer_normalDepth[neighborIndex];
			float depthQ = neighborND.w;

			if (depthQ < 0.0f) {
				// Neighbor hit environment map, discard
				continue;
			}

			glm::vec3 normalQ = glm::vec3(neighborND);
			glm::vec4 colorQ = dev_inputColor[neighborIndex];
			float luminanceQ = computeLuminance(glm::vec3(colorQ));

			float h_q = h[r + 2] * h[c + 2];

			float w_pq = computeTotalWeight(
				p, q,
				depthP, depthQ, sigmaDepth, depthGradP,
				normalP, normalQ, sigmaNormal,
				luminanceP, luminanceQ, prefilteredVarianceP, sigmaLuminance
			);

			sumColor += h_q * w_pq * glm::vec3(colorQ);
			sumColorWeight += h_q * w_pq;

			sumVariance += h_q * h_q * w_pq * w_pq * varianceP;
			sumVarianceWeight += h_q * w_pq;
		}
	}

	if (sumColorWeight > 0.0f) {
		dev_outputColor[pixelIndex] = glm::vec4(sumColor / sumColorWeight, colorP.w);
	}
	else {
		dev_outputColor[pixelIndex] = colorP;
	}

	if (sumVarianceWeight > 0.0f) {
		dev_outputVariance[pixelIndex] = sumVariance / (sumVarianceWeight * sumVarianceWeight);
	}
	else {
		dev_outputVariance[pixelIndex] = varianceP;
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

__global__ void kernDebugSVGFIllumination(
	const glm::vec4* dev_pingBuffer,
	cudaSurfaceObject_t surface,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;

	// Read the accumulated HDR illumination from SVGF
	glm::vec3 hdrColor = glm::vec3(dev_pingBuffer[pixelIndex]);

	// Apply simple Gamma Correction (2.2) and clamp to visible bounds
	glm::vec3 finalColor;
	finalColor.x = glm::clamp(powf(hdrColor.x, 1.0f / 2.2f), 0.0f, 1.0f);
	finalColor.y = glm::clamp(powf(hdrColor.y, 1.0f / 2.2f), 0.0f, 1.0f);
	finalColor.z = glm::clamp(powf(hdrColor.z, 1.0f / 2.2f), 0.0f, 1.0f);

	// Pack into the uchar4 format your display surface expects
	uchar4 pixelColor;
	pixelColor.x = (unsigned char)(finalColor.z * 255.0f); // Blue
	pixelColor.y = (unsigned char)(finalColor.y * 255.0f); // Green
	pixelColor.z = (unsigned char)(finalColor.x * 255.0f); // Red
	pixelColor.w = 255;

	surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
}

__global__ void kernDebugVariance(
	const float* dev_variance,
	cudaSurfaceObject_t surface,
	int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) return;

	int pixelIndex = y * width + x;
	float variance = dev_variance[pixelIndex];

	// Amplify variance visually since raw statistical variance values are tiny decimal numbers
	float visualIntensity = glm::clamp(variance * 10.0f, 0.0f, 1.0f);

	uchar4 pixelColor;
	pixelColor.x = (unsigned char)(visualIntensity * 255.0f); // Blue
	pixelColor.y = 0;                                         // Green
	pixelColor.z = (unsigned char)(visualIntensity * 255.0f); // Red (Creates Magenta for noise)
	pixelColor.w = 255;

	surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
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
		dev_gBuffer_historyLengthPrev = other.dev_gBuffer_historyLengthPrev;
		dev_illuminationPrev = other.dev_illuminationPrev;
		dev_pingBuffer = other.dev_pingBuffer;
		dev_pongBuffer = other.dev_pongBuffer;
		dev_moments = other.dev_moments;
		dev_momentsPrev = other.dev_momentsPrev;
		dev_variancePing = other.dev_variancePing;
		dev_variancePong = other.dev_variancePong;
		dev_prefilteredVariance = other.dev_prefilteredVariance;

		other.m_width = 0;
		other.m_height = 0;
		other.dev_gBuffer_normalDepth = nullptr;
		other.dev_gBuffer_motionVectors = nullptr;
		other.dev_gBuffer_historyLength = nullptr;
		other.dev_gBuffer_normalDepthPrev = nullptr;
		other.dev_gBuffer_historyLengthPrev = nullptr;
		other.dev_illuminationPrev = nullptr;
		other.dev_pingBuffer = nullptr;
		other.dev_pongBuffer = nullptr;
		other.dev_moments = nullptr;
		other.dev_momentsPrev = nullptr;
		other.dev_variancePing = nullptr;
		other.dev_variancePong = nullptr;
		other.dev_prefilteredVariance = nullptr;
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
	std::swap(dev_gBuffer_historyLength, dev_gBuffer_historyLengthPrev);

	size_t numPixels = m_width * m_height;
	cudaMemcpy(dev_illuminationPrev, dev_pingBuffer, numPixels * sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
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

void SVGFManager::executeTemporalAccumulation()
{
	if (m_width <= 0 || m_height <= 0) {
		return;
	}

	dim3 blockSize(16, 16);
	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);

	kernTemporalAccumulation << <gridSize, blockSize >> > (
		dev_gBuffer_normalDepth,
		dev_gBuffer_motionVectors,
		dev_gBuffer_normalDepthPrev,
		dev_pingBuffer,
		dev_illuminationPrev,
		dev_momentsPrev,
		// Outputs
		dev_pongBuffer,
		dev_moments,
		dev_gBuffer_historyLength,
		dev_gBuffer_historyLengthPrev,
		m_width,
		m_height
		);

	// Swap ping and pong buffers so we can use this output as input for future operations
	std::swap(dev_pingBuffer, dev_pongBuffer);
}

void SVGFManager::executeVarianceEstimation()
{
	if (m_width <= 0 || m_height <= 0) {
		return;
	}

	dim3 blockSize(16, 16);
	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);

	kernEstimateVariance << <gridSize, blockSize >> > (
		dev_pingBuffer,
		dev_moments,
		dev_gBuffer_normalDepth,
		dev_gBuffer_historyLength,
		dev_variancePing,
		m_width,
		m_height
		);
}

void SVGFManager::executeAtrousFilteringPipeline() {
	if (m_width <= 0 || m_height <= 0) return;

	dim3 blockSize(16, 16);
	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);

	// SVGF Tuning coefficients
	float sigmaLuminance = 4.0f;
	float sigmaNormal = 128.0f;
	float sigmaDepth = 1.0f;

	const int iterations = 5;

	for (int i = 0; i < iterations; ++i) {
		int stride = 1 << i;

		// Prefilter variance
		kernVariancePrefilter3x3 << <gridSize, blockSize >> > (
			dev_variancePing,
			dev_gBuffer_normalDepth,
			dev_prefilteredVariance,
			m_width, m_height
			);

		kernAtrousFilter << <gridSize, blockSize >> > (
			dev_pingBuffer,
			dev_gBuffer_normalDepth,
			dev_variancePing,
			dev_prefilteredVariance,
			dev_variancePong,
			dev_pongBuffer,
			stride,
			sigmaLuminance,
			sigmaNormal,
			sigmaDepth,
			m_width,
			m_height
			);

		// Ping pong
		std::swap(dev_pingBuffer, dev_pongBuffer);
		std::swap(dev_variancePing, dev_variancePong);
	}
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

void SVGFManager::debugIlluminance(cudaSurfaceObject_t surface)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);

	kernDebugSVGFIllumination << <gridSize, blockSize >> > (dev_pingBuffer, surface, m_width, m_height);
}

void SVGFManager::debugVariance(cudaSurfaceObject_t surface)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x, (m_height + blockSize.y - 1) / blockSize.y);
	kernDebugVariance << <gridSize, blockSize >> > (dev_variancePing, surface, m_width, m_height);
}

void SVGFManager::allocateBuffers()
{
	size_t numPixels = static_cast<size_t>(m_width) * m_height;

	CUDA_CHECK(cudaMalloc(&dev_gBuffer_normalDepth, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_normalDepthPrev, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_motionVectors, numPixels * sizeof(glm::vec2)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_historyLength, numPixels * sizeof(unsigned int)));
	CUDA_CHECK(cudaMalloc(&dev_gBuffer_historyLengthPrev, numPixels * sizeof(unsigned int)));

	CUDA_CHECK(cudaMalloc(&dev_illuminationPrev, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_pingBuffer, numPixels * sizeof(glm::vec4)));
	CUDA_CHECK(cudaMalloc(&dev_pongBuffer, numPixels * sizeof(glm::vec4)));

	CUDA_CHECK(cudaMalloc(&dev_moments, numPixels * sizeof(glm::vec2)));
	CUDA_CHECK(cudaMalloc(&dev_momentsPrev, numPixels * sizeof(glm::vec2)));
	CUDA_CHECK(cudaMalloc(&dev_variancePing, numPixels * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dev_variancePong, numPixels * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dev_prefilteredVariance, numPixels * sizeof(float)));

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

	if (dev_gBuffer_historyLengthPrev) {
		cudaFree(dev_gBuffer_historyLengthPrev);
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

	if (dev_variancePing) {
		cudaFree(dev_variancePing);
	}

	if (dev_variancePong) {
		cudaFree(dev_variancePong);
	}

	if (dev_prefilteredVariance) {
		cudaFree(dev_prefilteredVariance);
	}

	dev_gBuffer_normalDepth = nullptr; 
	dev_gBuffer_normalDepthPrev = nullptr;
	dev_gBuffer_motionVectors = nullptr; 
	dev_gBuffer_historyLength = nullptr;
	dev_gBuffer_historyLengthPrev = nullptr;
	dev_illuminationPrev = nullptr;
	dev_pingBuffer = nullptr; 
	dev_pongBuffer = nullptr;
	dev_moments = nullptr; 
	dev_momentsPrev = nullptr;
	dev_variancePing = nullptr;
	dev_variancePong = nullptr;
	dev_prefilteredVariance = nullptr;
}