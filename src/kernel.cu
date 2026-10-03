#include "kernel.h"

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/partition.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/sort.h>

#include "samplers.h"
#include "material.h"
#include "intersection.h"

#define MIN_RUSSIAN_ROULETTE_BOUNCES 2

__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}


__host__ __device__ int divup(int a, int b) {
    return (a + b - 1) / b;
}

__device__ void get2DIndex(int index1D, int width, int* outX, int* outY) {
    *outX = index1D % width;
    *outY = index1D / width;
}

__device__ glm::vec3 reinhardToneMap(const glm::vec3& color) {
    return color / (color + glm::vec3(1.0f));
}

__device__ glm::vec3 gammaCorrect(const glm::vec3& color, float gamma) {
    return glm::clamp(glm::pow(color, glm::vec3(1.0f / gamma)), 0.0f, 1.0f);
}

__device__ uchar4 convertColorToUChar4(const glm::vec3& color) {
    uchar4 pixelColor;
    pixelColor.x = (unsigned char)(color.z * 255.0f); // Blue
    pixelColor.y = (unsigned char)(color.y * 255.0f); // Green
    pixelColor.z = (unsigned char)(color.x * 255.0f); // Red
    pixelColor.w = 255; // Alpha
    return pixelColor;
}

__device__ void addRadiance(PathState& pathState, const glm::vec3& radiance, bool bDirect) {
    // Direct lighting is anything coming directly from a light source, so
    // either NEE at bounce 0, or bounce 1
    pathState.accumulatedColor += radiance;
    if (bDirect) {
        pathState.directColor += radiance;
    }
}

__device__ void writePathStateToSurface(const PathState& pathState, 
    cudaSurfaceObject_t surface, 
    glm::vec3* dev_accumulatedColor,
    glm::vec4* dev_currentDirectColor,
    glm::vec4* dev_currentIndirectColor,
    unsigned int* dev_sampleCounts, 
    int width,
    bool bHit) {
    int pixelIndex = pathState.pixelIndex;

    unsigned int currentSampleIndex = atomicAdd(&dev_sampleCounts[pixelIndex], 1);

    dev_accumulatedColor[pixelIndex] += pathState.accumulatedColor;
    if (dev_currentDirectColor != nullptr && dev_currentIndirectColor != nullptr) {
        // Record direct and indirect color for svgf filtering
        float hitValue = bHit ? 1.0f : 0.0f;
        dev_currentDirectColor[pixelIndex] = glm::vec4(pathState.directColor, hitValue);
        dev_currentIndirectColor[pixelIndex] = glm::vec4(pathState.accumulatedColor - pathState.directColor, hitValue);
    }

    // Get average color over all samples
    glm::vec3 averageColor = dev_accumulatedColor[pixelIndex] / (float)(currentSampleIndex + 1);

	glm::vec3 tonemappedColor = reinhardToneMap(averageColor);

	glm::vec3 finalColor = gammaCorrect(tonemappedColor, 2.2f);

	uchar4 pixelColor = convertColorToUChar4(finalColor);

    int pixelIndexX = 0;
    int pixelIndexY = 0;
    get2DIndex(pixelIndex, width, &pixelIndexX, &pixelIndexY);

    surf2Dwrite(pixelColor, surface, pixelIndexX * sizeof(uchar4), pixelIndexY);
}

__device__ glm::vec3 sampleEnvironmentMap(cudaTextureObject_t environmentMap, const glm::vec3& direction) {
    if (environmentMap == 0) {
        return glm::vec3(0.0f);
    }

    float theta = glm::acos(direction.y);
    float phi = atan2f(direction.z, direction.x);

    float u = 1.0f - (phi + glm::pi<float>()) / (2.0f * glm::pi<float>());
    float v = theta / glm::pi<float>();

    float4 sampled = tex2D<float4>(environmentMap, u, v);
    return glm::vec3(sampled.x, sampled.y, sampled.z);
}

__device__ void applyCameraDepthOfField(Ray& ray, 
    const glm::vec2& random, 
    glm::vec3& cameraLook,
    glm::vec3& cameraRight,
    glm::vec3& cameraUp,
    float focalDistance, 
    float lensRadius) {
    glm::vec2 pLens = lensRadius * Samplers::sampleUniformDiskConcentric(random);

    float ft = focalDistance / glm::dot(ray.direction, cameraLook);
    glm::vec3 pFocus = ray.getPositionAtTime(ft);

    glm::vec3 worldLensOrigin = ray.origin + 
        cameraRight * pLens.x + 
        cameraUp * pLens.y;

    ray.origin = worldLensOrigin;
    ray.direction = glm::normalize(pFocus - ray.origin);
}

__global__ void kernGenerateCameraRays(PathState* dev_pathStates, 
    int width, 
    int height, 
    glm::vec3 cameraPos, 
    glm::vec3 cameraLook, 
    glm::vec3 cameraRight, 
    glm::vec3 cameraUp, 
    float fovY, 
    int frameIndex,
    bool bMSAAEnabled,
    float focalDistance,
    float lensRadius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // row major pixel indexing
    int pixelIndex = y * width + x;

    // Uncomment for more stable svgf
    float jitterX = 0.0f;
    float jitterY = 0.0f;

    thrust::default_random_engine rng = makeSeededRandomEngine(frameIndex, pixelIndex, 0);
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (bMSAAEnabled) {
        // Add antialiasing by jittering the ray
        jitterX = u01(rng) - 0.5f;
		jitterY = u01(rng) - 0.5f;
    }

    // Map to range [-1, 1]
    float normalizedX = (2.0f * (x + 0.5f + jitterX) / (float)width) - 1.0f;
    float normalizedY = 1.0f - (2.0f * (y + 0.5f + jitterY) / (float)height);

    float aspectRatio = (float)width / (float)height;
    float scale = tanf(glm::radians(fovY) * 0.5f);

    glm::vec3 rayDirection = cameraLook + (normalizedX * scale * aspectRatio * cameraRight) +
        (normalizedY * scale * cameraUp);
    rayDirection = glm::normalize(rayDirection);


    PathState pathState;
    pathState.ray.origin = cameraPos;
    pathState.ray.direction = rayDirection;
    pathState.pixelIndex = pixelIndex;
    pathState.throughput = glm::vec3(1.0f);
    pathState.accumulatedColor = glm::vec3(0.0f);
    pathState.directColor = glm::vec3(0.0f);
    pathState.bounceCount = 0;
    pathState.active = true;

    if (lensRadius > 0.001f) {
        applyCameraDepthOfField(pathState.ray, glm::vec2(u01(rng), u01(rng)), cameraLook, cameraRight, cameraUp, focalDistance, lensRadius);
    }

    dev_pathStates[pixelIndex] = pathState;
}

__global__ void kernIntersect(PathState* dev_pathStates, IntersectionData* dev_intersectionData,
    int activePathCount,
    DevScene dev_scene)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= activePathCount) {
        return;
    }

    // Stream compaction should catch this
    if (!dev_pathStates[index].active) {
        return;
    }

    Ray currentRay = dev_pathStates[index].ray;

    IntersectionData closestIntersection = dev_scene.intersect(currentRay);

    dev_intersectionData[index] = closestIntersection;
}

__global__ void kernShade(
    PathState* dev_pathStates,
    IntersectionData* dev_intersectionData,
    DevScene dev_scene,
    cudaTextureObject_t environmentMap,
    int activePathCount,
    cudaSurfaceObject_t surface,
    glm::vec3* dev_accumulatedColor,
    glm::vec4* dev_currentDirectColor,
    glm::vec4* dev_currentIndirectColor,
    unsigned int* dev_sampleCounts,
    int width,
    int iteration,
    int frameIndex)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= activePathCount) {
        return;
    }

    // Stream compaction should catch this
    if (!dev_pathStates[index].active) {
        return;
    }

    IntersectionData intersectionData = dev_intersectionData[index];
    PathState& pathState = dev_pathStates[index];

    // Handle misses
    if (intersectionData.t <= 0.0f) {
        // Considered direct if from the camera or off of the first bounce
        addRadiance(pathState, pathState.throughput * sampleEnvironmentMap(environmentMap, pathState.ray.direction), pathState.bounceCount <= 1);
        pathState.active = false;
        writePathStateToSurface(pathState, surface, dev_accumulatedColor, dev_currentDirectColor, dev_currentIndirectColor, dev_sampleCounts, width, pathState.bounceCount == 0);
        return;
    }

    // Handle invalid material (shouldn't happen)
    if (intersectionData.materialIndex >= dev_scene.m_materialCount) {
        pathState.active = false;
        return;
    }

    // Make copy of material so we can load in roughness and albedo from textures
    Material material = dev_scene.dev_materials[intersectionData.materialIndex];
	material.initializeFromIntersection(intersectionData); // Set albedo, roughness, and metallic from textures if applicable

    // Handle hitting a light
    if (material.emittance > 0.0f) {
        // What are the chances that this would be hit by sampling the lights previously 
        float lightPdf = intersectionData.pdfIfLight;

        float misWeight = 1.0f;
        if (!pathState.previousDelta && (pathState.previousBrdfPdf + lightPdf) > 0.0f) {
			misWeight = MathHelpers::powerHeuristic(pathState.previousBrdfPdf, lightPdf);
        }

		// Considered direct if from the camera or off of the first bounce
        addRadiance(pathState, pathState.throughput * material.albedo * material.emittance * misWeight, pathState.bounceCount <= 1);

        // Hitting a light terminates the path
        pathState.active = false;
        writePathStateToSurface(pathState, surface, dev_accumulatedColor, dev_currentDirectColor, dev_currentIndirectColor, dev_sampleCounts, width, true);
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(frameIndex, index, iteration);
    thrust::uniform_real_distribution<float> u01(0, 1);
    
    // Next Event Estimation (Sample lights)
    if (!material.isDelta()) {
        glm::vec4 neeRandom = glm::vec4(u01(rng), u01(rng), u01(rng), u01(rng));
		glm::vec3 directionToLight = glm::vec3(0.0f);
        float lightPdf = 0.0f;

        glm::vec3 lightSample = dev_scene.nextEventEsimation(neeRandom, pathState.ray, intersectionData, directionToLight, lightPdf);

        if (lightPdf > 0.0f && glm::length(lightSample) > 0.0f) {
			// Find the pdf if this direction was sampled from the BRDF
			glm::vec3 wi = -pathState.ray.direction;
			glm::vec3 f = material.evaluate(intersectionData.normal, wi, directionToLight, intersectionData.bInside, false);
			float pdfBrdf = material.pdf(intersectionData.normal, wi, directionToLight, intersectionData.bInside, false);

			float misWeight = MathHelpers::powerHeuristic(lightPdf, pdfBrdf);

            float cosThetaNE = glm::max(0.0f, glm::abs(glm::dot(intersectionData.normal, directionToLight)));
            // Considered direct if we are doing NEE off of the first bounce
			addRadiance(pathState, pathState.throughput * (lightSample / lightPdf) * f * cosThetaNE * misWeight, pathState.bounceCount == 0);
        }
    }

	// Indirect lighting (Sample BRDF)
    glm::vec3 wi = -pathState.ray.direction;
    glm::vec3 wo = glm::vec3(0.0f);
    glm::vec3 sampleThroughput = glm::vec3(0.0f);
    float brdfPdf = 0.0f;
    bool bIsTransmission = false;

	material.sample(
        intersectionData.normal, 
        wi, 
        glm::vec3(u01(rng), u01(rng), u01(rng)), 
        intersectionData.bInside, 
        wo, 
        sampleThroughput, 
        brdfPdf, 
        bIsTransmission);

    // Update path state history tracking
	pathState.previousBrdfPdf = brdfPdf;
	pathState.previousDelta = material.isDelta();

    // Handle invalid pdf or throughput
	if (brdfPdf <= 0.0f || glm::all(glm::lessThanEqual(sampleThroughput, glm::vec3(0.0f)))) {
        pathState.active = false;
        writePathStateToSurface(pathState, surface, dev_accumulatedColor, dev_currentDirectColor, dev_currentIndirectColor, dev_sampleCounts, width, true);
        return;
	}

    pathState.throughput *= sampleThroughput;

    // Update ray for next iteration
    glm::vec3 intersectionPosition = pathState.ray.getPositionAtTime(intersectionData.t);
    pathState.ray = Ray::generateBouncedRay(intersectionData.normal, intersectionPosition, wo);

    // Run russian roulette
    if (pathState.bounceCount >= MIN_RUSSIAN_ROULETTE_BOUNCES) {
        float survivalProbability = glm::max(pathState.throughput.x, glm::max(pathState.throughput.y, pathState.throughput.z));
        survivalProbability = glm::clamp(survivalProbability, 0.05f, 0.95f);

        if (u01(rng) > survivalProbability) {
            // Terminate path
            pathState.active = false;
            writePathStateToSurface(pathState, surface, dev_accumulatedColor, dev_currentDirectColor, dev_currentIndirectColor, dev_sampleCounts, width, true);
            return;
        }

        // Compensate for survival probability
        pathState.throughput /= survivalProbability;
    }

    // Increment bounce count
    pathState.bounceCount += 1;
}

__global__ void kernDebugUV(cudaSurfaceObject_t surface,
    IntersectionData* intersectionData,
    int n,
    int width)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int pixelIndex = y * width + x;
    if(pixelIndex < 0 || pixelIndex >= n) {
        return;
	}
	IntersectionData intersection = intersectionData[pixelIndex];

    float r = intersection.uv.x;
    float g = intersection.uv.y;
    float b = 0.0f;

    uchar4 pixelColor;
    pixelColor.x = (unsigned char)(b * 255.0f); // Blue
    pixelColor.y = (unsigned char)(g * 255.0f); // Green
    pixelColor.z = (unsigned char)(r * 255.0f); // Red
    pixelColor.w = 255;

    surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
}

__global__ void kernColorSurface(cudaSurfaceObject_t surface, 
    glm::vec3* dev_accumulatedColor,
    glm::vec4* dev_currentDirectColor,
    glm::vec4* dev_currentIndirectColor,
    unsigned int* dev_sampleCounts, 
    PathState* dev_pathStates, 
    int n, 
    int width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    PathState currentPathState = dev_pathStates[index];

    writePathStateToSurface(currentPathState, surface, dev_accumulatedColor, dev_currentDirectColor, dev_currentIndirectColor, dev_sampleCounts, width, true);
}

__global__ void kernDebugRays(PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int pixelIndex = y * width + x;
    PathState pathState = dev_pathStates[pixelIndex];
    Ray ray = pathState.ray;


    float r = ray.direction.x * 0.5f + 0.5f;
    float g = ray.direction.y * 0.5f + 0.5f;
    float b = ray.direction.z * 0.5f + 0.5f;
    
    uchar4 pixelColor;
    pixelColor.x = (unsigned char)(b * 255.0f); // Blue
    pixelColor.y = (unsigned char)(g * 255.0f); // Green
    pixelColor.z = (unsigned char)(r * 255.0f); // Red
    pixelColor.w = 255;

    surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
}

__global__ void fillSurfaceColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    uchar4 pixelColor;
    pixelColor.x = (unsigned char)(b * 255.0f); // Blue
    pixelColor.y = (unsigned char)(g * 255.0f); // Green
    pixelColor.z = (unsigned char)(r * 255.0f); // Red
    pixelColor.w = 255;                         // Alpha

    surf2Dwrite(pixelColor, surface, x * sizeof(uchar4), y);
}

void launchCameraRayGenKernel(
    PathState* dev_pathStates, 
    int width, int height, 
    const Camera& camera, 
    int frameIndex,
    bool bMSAAEnabled)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    kernGenerateCameraRays << <gridSize, blockSize >> > (
        dev_pathStates, width, height,
        camera.m_position, camera.m_look, camera.m_right, camera.m_up, camera.m_fovy, 
        frameIndex,
        bMSAAEnabled,
        camera.m_focalDistance, camera.m_lensRadius);
}

void launchIntersectKernel(PathState* dev_pathStates, IntersectionData* dev_intersectionData, const std::unique_ptr<Scene>& scene, int activePathCount)
{
    dim3 blockSize(32);
    dim3 gridSize(divup(activePathCount, blockSize.x));

    kernIntersect << <gridSize, blockSize >> > (dev_pathStates, 
        dev_intersectionData, 
        activePathCount, 
        scene != nullptr ? scene->getDevScene() : DevScene{});
}

void launchShadeKernel(PathState* dev_pathStates,
    IntersectionData* dev_intersectionData,
    const std::unique_ptr<Scene>& scene,
    const std::unique_ptr<EnvironmentMap>& environmentMap,
    int activePathCount,
    cudaSurfaceObject_t surface,
    glm::vec3* dev_accumulatedColor,
    glm::vec4* dev_currentDirectColor,
    glm::vec4* dev_currentIndirectColor,
    unsigned int* dev_sampleCounts,
    int width,
    int iteration,
    int frameIndex)
{
    dim3 blockSize(32);
    dim3 gridSize(divup(activePathCount, blockSize.x));

    kernShade << <gridSize, blockSize >> > (dev_pathStates,
        dev_intersectionData,
        scene != nullptr ? scene->getDevScene() : DevScene{},
        environmentMap != nullptr ? environmentMap->m_environmentMapTexture : 0,
        activePathCount,
        surface,
        dev_accumulatedColor,
        dev_currentDirectColor,
        dev_currentIndirectColor,
        dev_sampleCounts,
        width,
        iteration,
        frameIndex);
}

void launchColorSurfaceKernel(PathState* dev_pathStates,
    int activePathCount,
    cudaSurfaceObject_t surface,
    glm::vec3* dev_accumulatedColor,
    glm::vec4* dev_currentDirectColor,
    glm::vec4* dev_currentIndirectColor,
    unsigned int* dev_sampleCounts,
    int width)
{
    dim3 blockSize(32);
    dim3 gridSize(divup(activePathCount, blockSize.x));

    kernColorSurface<<<gridSize, blockSize>>>(surface, 
        dev_accumulatedColor,
        dev_currentDirectColor,
        dev_currentIndirectColor,
        dev_sampleCounts,
        dev_pathStates, 
        activePathCount, 
        width);
}

void launchDebugRaysKernel( PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    kernDebugRays << <gridSize, blockSize >> > (dev_pathStates, surface, width, height);
}

void launchColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    fillSurfaceColorKernel << <gridSize, blockSize >> > (surface, width, height, r, g, b);
}

void launchDebugUVKernel(IntersectionData* dev_intersectionData, cudaSurfaceObject_t surface, int width, int height)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	kernDebugUV << <gridSize, blockSize >> > (surface, dev_intersectionData, width * height, width);
}

struct is_path_active {
    __host__ __device__ bool operator()(const thrust::tuple<PathState, IntersectionData>& tuple) {
        return thrust::get<0>(tuple).active;
    }
};

int runStreamCompaction(PathState* dev_pathStates, IntersectionData* dev_intersectionData, int numActivePaths) {
    thrust::device_ptr<PathState> th_path_start(dev_pathStates);
    thrust::device_ptr<IntersectionData> th_inter_start(dev_intersectionData);

    auto zip_start = thrust::make_zip_iterator(thrust::make_tuple(th_path_start, th_inter_start));
    auto zip_end = zip_start + numActivePaths;

    // stable_partition moves active paths to the front, preserving relative order
    auto zip_new_end = thrust::stable_partition(
        thrust::device,
        zip_start,
        zip_end,
        is_path_active()
    );

    return zip_new_end - zip_start;
}


struct GetMaterialType {
    const Material* dev_materials;

    GetMaterialType(const Material* materials) : dev_materials(materials) {}

    __device__ uint8_t operator()(const IntersectionData& intersection) const {
        if (intersection.t <= 0.0f || dev_materials == nullptr) {
            return 0;
        }

        const Material& material = dev_materials[intersection.materialIndex];

        if (material.type == MaterialType::PerfectSpecular) {
            return 1;
        }

        if (material.type == MaterialType::BlinnPhong) {
            return 2;
        }

        return 3;
    }
};

void sortPathsByMaterial(PathState* dev_pathStates, IntersectionData* dev_intersectionData, int activePathCount, const Material* dev_materials)
{
    thrust::device_ptr<PathState> thrust_pathStates(dev_pathStates);
    thrust::device_ptr<IntersectionData> thrust_intersectionData(dev_intersectionData);

    uint8_t* dev_materialTypes = nullptr;
    int* dev_sortedIndices = nullptr;
    PathState* dev_pathStatesTemp = nullptr;
    IntersectionData* dev_intersectionDataTemp = nullptr;

    if (cudaMalloc((void**)&dev_materialTypes, activePathCount * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc((void**)&dev_sortedIndices, activePathCount * sizeof(int)) != cudaSuccess ||
        cudaMalloc((void**)&dev_pathStatesTemp, activePathCount * sizeof(PathState)) != cudaSuccess ||
        cudaMalloc((void**)&dev_intersectionDataTemp, activePathCount * sizeof(IntersectionData)) != cudaSuccess) {
        cudaFree(dev_materialTypes);
        cudaFree(dev_sortedIndices);
        cudaFree(dev_pathStatesTemp);
        cudaFree(dev_intersectionDataTemp);
        return;
    }

    thrust::device_ptr<uint8_t> thrust_materialTypes(dev_materialTypes);
    thrust::device_ptr<int> thrust_sortedIndices(dev_sortedIndices);

	// Get material types for each intersection
    GetMaterialType transformFunction(dev_materials);
    thrust::transform(thrust_intersectionData, thrust_intersectionData + activePathCount, thrust_materialTypes, transformFunction);

    // Do a sort of indices based on material types to do a scatter of path states and intersection data
    thrust::sequence(thrust_sortedIndices, thrust_sortedIndices + activePathCount);
    thrust::sort_by_key(thrust_materialTypes, thrust_materialTypes + activePathCount, thrust_sortedIndices);

    // Reorder paths and intersections by the sorted permutation
    auto zip_values = thrust::make_zip_iterator(thrust::make_tuple(thrust_pathStates, thrust_intersectionData));
    auto zip_temp = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<PathState>(dev_pathStatesTemp),
        thrust::device_ptr<IntersectionData>(dev_intersectionDataTemp)));
    // Gather into temporary arrays
    thrust::gather(thrust_sortedIndices, thrust_sortedIndices + activePathCount, zip_values, zip_temp);

    // Copy sorted data back to original arrays
    cudaMemcpy(dev_pathStates, dev_pathStatesTemp, activePathCount * sizeof(PathState), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_intersectionData, dev_intersectionDataTemp, activePathCount * sizeof(IntersectionData), cudaMemcpyDeviceToDevice);

    cudaFree(dev_materialTypes);
    cudaFree(dev_sortedIndices);
    cudaFree(dev_pathStatesTemp);
    cudaFree(dev_intersectionDataTemp);
}
