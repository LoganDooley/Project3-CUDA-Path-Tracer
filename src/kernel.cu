#include "kernel.h"

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/partition.h>

#include "samplers.h"
#include "shadingMaterial.h"

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

__device__ void writePathStateToSurface(const PathState& pathState, 
    cudaSurfaceObject_t surface, 
    glm::vec3* dev_accumulatedColor,
    unsigned int* dev_sampleCounts, 
    int width) {
    int pixelIndex = pathState.pixelIndex;

    unsigned int currentSampleIndex = atomicAdd(&dev_sampleCounts[pixelIndex], 1);

    dev_accumulatedColor[pixelIndex] += pathState.accumulatedColor;

    // Get average color over all samples
    glm::vec3 averageColor = dev_accumulatedColor[pixelIndex] / (float)(currentSampleIndex + 1);

    glm::vec3 finalColor;
    finalColor.x = glm::clamp(powf(averageColor.x, 1.0f / 2.2f), 0.0f, 1.0f);
    finalColor.y = glm::clamp(powf(averageColor.y, 1.0f / 2.2f), 0.0f, 1.0f);
    finalColor.z = glm::clamp(powf(averageColor.z, 1.0f / 2.2f), 0.0f, 1.0f);

    uchar4 pixelColor;
    pixelColor.x = (unsigned char)(finalColor.z * 255.0f); // Blue
    pixelColor.y = (unsigned char)(finalColor.y * 255.0f); // Green
    pixelColor.z = (unsigned char)(finalColor.x * 255.0f); // Red
    pixelColor.w = 255;

    int pixelIndexX = 0;
    int pixelIndexY = 0;
    get2DIndex(pixelIndex, width, &pixelIndexX, &pixelIndexY);

    surf2Dwrite(pixelColor, surface, pixelIndexX * sizeof(uchar4), pixelIndexY);
}

__global__ void kernGenerateCameraRays(PathState* dev_pathStates, 
    int width, 
    int height, 
    glm::vec3 cameraPos, 
    glm::vec3 cameraLook, 
    glm::vec3 cameraRight, 
    glm::vec3 cameraUp, 
    float fovY, 
    int frameIndex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // row major pixel indexing
    int pixelIndex = y * width + x;

    // Add antialiasing by jittering the ray
    thrust::default_random_engine rng = makeSeededRandomEngine(frameIndex, pixelIndex, 0);
    thrust::uniform_real_distribution<float> u01(0, 1);

    float jitterX = u01(rng) - 0.5f;
    float jitterY = u01(rng) - 0.5f;

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
    pathState.bounceCount = 0;
    pathState.active = true;

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
    int activePathCount,
    cudaSurfaceObject_t surface,
    glm::vec3* dev_accumulatedColor,
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

    if (intersectionData.t <= 0.0f) {
        // Add environment lighting & mark terminated
        //dev_pathStates[index].accumulatedColor += dev_pathStates[index].throughput * glm::vec3(0.0f, 0.3f, 0.7f);
        pathState.active = false;
        writePathStateToSurface(pathState, surface, dev_accumulatedColor, dev_sampleCounts, width);
        return;
    }

    if (intersectionData.materialIndex >= dev_scene.m_materialCount) {
        pathState.active = false;
        return;
    }

    Material material = dev_scene.dev_materials[intersectionData.materialIndex];

    // Add emissive
    pathState.accumulatedColor += pathState.throughput * material.color * material.emittance;

    thrust::default_random_engine rng = makeSeededRandomEngine(iteration, index, frameIndex);
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Run russian roulette
    if (pathState.bounceCount >= MIN_RUSSIAN_ROULETTE_BOUNCES) {
        float survivalProbability = glm::max(pathState.throughput.x, glm::max(pathState.throughput.y, pathState.throughput.z));
        survivalProbability = glm::clamp(survivalProbability, 0.05f, 0.95f);

        if (u01(rng) > survivalProbability) {
            // Terminate path
            pathState.active = false;
            writePathStateToSurface(pathState, surface, dev_accumulatedColor, dev_sampleCounts, width);
            return;
        }

        // Compensate for survival probability
        pathState.throughput /= survivalProbability;
    }

    // Generate diffuse ray direction
    glm::vec2 random = glm::vec2(u01(rng), u01(rng));
    glm::vec3 outgoingDirection = glm::vec3(0.0f);
    glm::vec3 brdfWeight = glm::vec3(0.0f);
    glm::vec3 wi = -pathState.ray.direction;

    float epsilonSign = 1.0f;

    bool bUseReflectRefract = (material.hasReflective > 0.0f || material.hasRefractive > 0.0f);

    bool bUseSpecular = (material.specular.color.x > 0.0f || material.specular.color.y > 0.0f || material.specular.color.z > 0.0f);

    if (bUseReflectRefract) {
        brdfWeight = ShadingMaterial::evaluatePerfectSpecularMaterial(material,
            intersectionData.normal,
            wi,
            random.x,
            intersectionData.bInside,
            outgoingDirection,
            epsilonSign);
    }
    else if (bUseSpecular) {
        brdfWeight = ShadingMaterial::evaluateGlossySpecularMaterial(material,
            intersectionData.normal,
            wi,
            random,
            outgoingDirection);
    }
    else {
        brdfWeight = ShadingMaterial::evaluateDiffuseMaterial(material,
            intersectionData.normal,
            random,
            outgoingDirection);
    }

    // Did brdf return a valid value
    if (brdfWeight.x <= 0.0f && brdfWeight.y <= 0.0f && brdfWeight.z <= 0.0f) {
        pathState.active = false;
        return;
    }

    pathState.throughput *= brdfWeight;

    // Update ray for next iteration
    const float EPSILON = 0.001f;
    pathState.ray.origin = pathState.ray.getPositionAtTime(intersectionData.t) + intersectionData.normal * epsilonSign * EPSILON;
    pathState.ray.direction = outgoingDirection;

    // Increment bounce count
    pathState.bounceCount += 1;
}

__global__ void kernColorSurface(cudaSurfaceObject_t surface, 
    glm::vec3* dev_accumulatedColor,
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

    writePathStateToSurface(currentPathState, surface, dev_accumulatedColor, dev_sampleCounts, width);
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

void launchCameraRayGenKernel(PathState* dev_pathStates, int width, int height, const Camera& camera, int frameIndex)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    kernGenerateCameraRays << <gridSize, blockSize >> > (
        dev_pathStates, width, height,
        camera.m_position, camera.m_look, camera.m_right, camera.m_up, camera.m_fovy, 
        frameIndex);
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
    int activePathCount,
    cudaSurfaceObject_t surface,
    glm::vec3* dev_accumulatedColor,
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
        activePathCount,
        surface,
        dev_accumulatedColor,
        dev_sampleCounts,
        width,
        iteration,
        frameIndex);
}

void launchColorSurfaceKernel(PathState* dev_pathStates,
    int activePathCount,
    cudaSurfaceObject_t surface,
    glm::vec3* dev_accumulatedColor,
    unsigned int* dev_sampleCounts,
    int width)
{
    dim3 blockSize(32);
    dim3 gridSize(divup(activePathCount, blockSize.x));

    kernColorSurface<<<gridSize, blockSize>>>(surface, 
        dev_accumulatedColor,
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