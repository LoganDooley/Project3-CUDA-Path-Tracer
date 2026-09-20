#include "kernel.h"

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>

#include "samplers.h"

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

__device__ void writePathStateToSurface(const PathState& pathState, cudaSurfaceObject_t surface, int width) {
    int pixelIndex = pathState.pixelIndex;

    int pixelIndexX = 0;
    int pixelIndexY = 0;
    get2DIndex(pixelIndex, width, &pixelIndexX, &pixelIndexY);

    uchar4 pixelColor;
    pixelColor.x = (unsigned char)(pathState.accumulatedColor.b * 255.0f); // Blue
    pixelColor.y = (unsigned char)(pathState.accumulatedColor.g * 255.0f); // Green
    pixelColor.z = (unsigned char)(pathState.accumulatedColor.r * 255.0f); // Red
    pixelColor.w = 255;

    surf2Dwrite(pixelColor, surface, pixelIndexX * sizeof(uchar4), pixelIndexY);
}

__global__ void kernGenerateCameraRays(PathState* dev_pathStates, int width, int height, glm::vec3 cameraPos, glm::vec3 cameraLook, glm::vec3 cameraRight, glm::vec3 cameraUp, float fovY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // row major pixel indexing
    int pixelIndex = y * width + x;

    // Map to range [-1, 1]
    float normalizedX = (2.0f * (x + 0.5f) / (float)width) - 1.0f;
    float normalizedY = 1.0f - (2.0f * (y + 0.5f) / (float)height);

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
    Geom* dev_geometry,
    int geometryCount)
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

    IntersectionData closestIntersection;
    for (int i = 0; i < geometryCount; i++) {
        IntersectionData intersection = IntersectionStatics::intersectGeometry(currentRay, dev_geometry[i]);
        if (intersection.t > 0.0f) {
            if (closestIntersection.t < 0.0f || intersection.t < closestIntersection.t) {
                closestIntersection = intersection;
            }
        }
    }

    dev_intersectionData[index] = closestIntersection;
}

__global__ void kernShade(
    PathState* dev_pathStates,
    IntersectionData* dev_intersectionData,
    Material* dev_materials,
    int materialCount,
    int activePathCount,
    cudaSurfaceObject_t surface,
    int width,
    int iteration)
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
    if (intersectionData.t <= 0.0f) {
        // Add environment lighting & mark terminated
        dev_pathStates[index].accumulatedColor += dev_pathStates[index].throughput * glm::vec3(0.0f, 0.3f, 0.7f);
        dev_pathStates[index].active = false;
        writePathStateToSurface(dev_pathStates[index], surface, width);
        return;
    }

    thrust::default_random_engine rng = makeSeededRandomEngine(iteration, index, 0);
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (intersectionData.materialIndex >= materialCount) {
        dev_pathStates[index].active = false;
        return;
    }

    Material material = dev_materials[intersectionData.materialIndex];

    // Add emissive
    dev_pathStates[index].accumulatedColor += dev_pathStates[index].throughput * material.color * material.emittance;

    // Generate diffuse ray direction
    glm::vec2 random = glm::vec2(u01(rng), u01(rng));
    glm::vec3 outgoingDirection = Samplers::sampleWorldUniformHemisphere(intersectionData.normal, random);
    
    // Compute cosTheta * brdf / pdf
    float cosTheta = glm::max(0.0f, glm::dot(intersectionData.normal, outgoingDirection));
    float pdf = 1.0f / (2.0f * glm::pi<float>());
    glm::vec3 brdf = material.color / glm::pi<float>();
    dev_pathStates[index].throughput *= (brdf * cosTheta) / pdf;

    // Update ray for next iteration
    const float EPSILON = 0.001f;
    dev_pathStates[index].ray.origin = dev_pathStates[index].ray.getPositionAtTime(intersectionData.t) + intersectionData.normal * EPSILON;
    dev_pathStates[index].ray.direction = outgoingDirection;
}

__global__ void kernColorSurface(cudaSurfaceObject_t surface, PathState* dev_pathStates, int n, int width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    PathState currentPathState = dev_pathStates[index];

    writePathStateToSurface(currentPathState, surface, width);
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

void launchCameraRayGenKernel(PathState* dev_pathStates, int width, int height, const Camera& camera)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    kernGenerateCameraRays << <gridSize, blockSize >> > (
        dev_pathStates, width, height,
        camera.m_position, camera.m_look, camera.m_right, camera.m_up, camera.m_fovy);
}

void launchIntersectKernel(PathState* dev_pathStates, IntersectionData* dev_intersectionData, Geom* dev_geometry,
    int geometryCount, int activePathCount)
{
    dim3 blockSize(32);
    dim3 gridSize(divup(activePathCount, blockSize.x));

    kernIntersect << <gridSize, blockSize >> > (dev_pathStates, dev_intersectionData, activePathCount, dev_geometry, geometryCount);
}

void launchShadeKernel(PathState* dev_pathStates,
    IntersectionData* dev_intersectionData,
    Material* dev_materials,
    int materialCount,
    int activePathCount,
    cudaSurfaceObject_t surface,
    int width,
    int iteration)
{
    dim3 blockSize(32);
    dim3 gridSize(divup(activePathCount, blockSize.x));

    kernShade << <gridSize, blockSize >> > (dev_pathStates, 
        dev_intersectionData, 
        dev_materials, 
        materialCount, 
        activePathCount,
        surface,
        width,
        iteration);
}

void launchColorSurfaceKernel(PathState* dev_pathStates,
    int activePathCount,
    cudaSurfaceObject_t surface,
    int width)
{
    dim3 blockSize(32);
    dim3 gridSize(divup(activePathCount, blockSize.x));

    kernColorSurface<<<gridSize, blockSize>>>(surface, dev_pathStates, activePathCount, width);
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

struct is_path_terminated {
    __host__ __device__
        bool operator()(const PathState& pathState) const {
        return !pathState.active;
    }
};

int runStreamCompaction(PathState* dev_pathStates, int numActivePaths) {
    thrust::device_ptr<PathState> th_pathStates_start(dev_pathStates);
    thrust::device_ptr<PathState> th_pathStates_end = th_pathStates_start + numActivePaths;

    thrust::device_ptr<PathState> th_new_end = thrust::remove_if(
        thrust::device,
        th_pathStates_start,
        th_pathStates_end,
        is_path_terminated()
    );

    return th_new_end - th_pathStates_start;
}