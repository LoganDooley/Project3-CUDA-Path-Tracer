#include "kernel.h"

__host__ __device__ int divup(int a, int b) {
    return (a + b - 1) / b;
}

__device__ void get2DIndex(int index1D, int width, int* outX, int* outY) {
    *outX = index1D % width;
    *outY = index1D / width;
}

__global__ void kernGenerateCameraRays(Ray* dev_rays, PathState* dev_pathStates, int width, int height, glm::vec3 cameraPos, glm::vec3 cameraLook, glm::vec3 cameraRight, glm::vec3 cameraUp, float fovY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // row major pixel indexing
    int pixelIndex = y * width + x;

    // Initialize path state
    dev_pathStates[pixelIndex].throughput = glm::vec3(1.0f);
    dev_pathStates[pixelIndex].accumulatedColor = glm::vec3(0.0f);
    dev_pathStates[pixelIndex].bounceCount = 0;
    dev_pathStates[pixelIndex].active = true;

    //// Map to range [-1, 1]
    float normalizedX = (2.0f * (x + 0.5f) / (float)width) - 1.0f;
    float normalizedY = 1.0f - (2.0f * (y + 0.5f) / (float)height);

    float aspectRatio = (float)width / (float)height;
    float scale = tanf(glm::radians(fovY) * 0.5f);

    glm::vec3 rayDirection = cameraLook + (normalizedX * scale * aspectRatio * cameraRight) +
        (normalizedY * scale * cameraUp);
    rayDirection = glm::normalize(rayDirection);

    Ray ray;
    ray.origin = cameraPos;
    ray.direction = rayDirection;
    ray.pixelIndex = pixelIndex;

    dev_rays[pixelIndex] = ray;

    PathState pathState = PathState{};
    dev_pathStates[pixelIndex] = pathState;
}

__global__ void kernIntersect(Ray* dev_rays,
    PathState* dev_pathStates,
    int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index > n) {
        return;
    }

    Ray currentRay = dev_rays[index];

    Sphere sceneSphere = Sphere{};
    IntersectionData intersection = sceneSphere.intersect(currentRay);

    if (intersection.bHit) {
        dev_pathStates[index].accumulatedColor = intersection.normal;
    }
}

__global__ void kernColorSurface(cudaSurfaceObject_t surface, Ray* dev_rays, PathState* dev_pathStates, int n, int width)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index > n) {
        return;
    }

    Ray currentRay = dev_rays[index];
    PathState currentPathState = dev_pathStates[index];

    int pixelIndex = currentRay.pixelIndex;

    int pixelIndexX = 0;
    int pixelIndexY = 0;
    get2DIndex(pixelIndex, width, &pixelIndexX, &pixelIndexY);

    uchar4 pixelColor;
    pixelColor.x = (unsigned char)(currentPathState.accumulatedColor.b * 255.0f); // Blue
    pixelColor.y = (unsigned char)(currentPathState.accumulatedColor.g * 255.0f); // Green
    pixelColor.z = (unsigned char)(currentPathState.accumulatedColor.r * 255.0f); // Red
    pixelColor.w = 255;

    surf2Dwrite(pixelColor, surface, pixelIndexX * sizeof(uchar4), pixelIndexY);
}

__global__ void kernDebugRays(Ray* dev_rays, PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int pixelIndex = y * width + x;
    Ray ray = dev_rays[pixelIndex];
    PathState pathState = dev_pathStates[pixelIndex];


    float r = ray.direction.x * 0.5f + 0.5f;
    float g = ray.direction.y * 0.5f + 0.5f;
    float b = ray.direction.z * 0.5f + 0.5f;

    r = pathState.accumulatedColor.r;
    g = pathState.accumulatedColor.g;
    b = pathState.accumulatedColor.b;
    
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

void launchCameraRayGenKernel(Ray* dev_rays, PathState* dev_pathStates, int width, int height, glm::vec3 cameraPos, glm::vec3 cameraLook, glm::vec3 cameraRight, glm::vec3 cameraUp, float fovY)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    kernGenerateCameraRays << <gridSize, blockSize >> > (
        dev_rays, dev_pathStates, width, height,
        cameraPos, cameraLook, cameraRight, cameraUp, fovY
        );
}

void launchIntersectKernel(Ray* dev_rays, PathState* dev_pathStates, int width, int height)
{
    int n = width * height;
    dim3 blockSize(32);
    dim3 gridSize(divup(n, blockSize.x));

    kernIntersect << <gridSize, blockSize >> > (dev_rays, dev_pathStates, width * height);
}

void launchColorSurfaceKernel(Ray* dev_rays, PathState* dev_pathStates, int width, int height, cudaSurfaceObject_t surface)
{
    int n = width * height;
    dim3 blockSize(32);
    dim3 gridSize(divup(n, blockSize.x));

    kernColorSurface<<<gridSize, blockSize>>>(surface, dev_rays, dev_pathStates, width * height, width);
}

void launchDebugRaysKernel(Ray* dev_rays, PathState* dev_pathStates, cudaSurfaceObject_t surface, int width, int height)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    kernDebugRays << <gridSize, blockSize >> > (dev_rays, dev_pathStates, surface, width, height);
}

void launchColorKernel(cudaSurfaceObject_t surface, int width, int height, float r, float g, float b) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    fillSurfaceColorKernel << <gridSize, blockSize >> > (surface, width, height, r, g, b);
}