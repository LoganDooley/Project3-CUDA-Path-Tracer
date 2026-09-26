#include "renderer.h"

#include "stb_image_write.h"

#include <vector>
#include <algorithm>
#include <cmath>

#include "kernel.h"

Renderer::Renderer() :
    m_svgfManager(std::make_unique<SVGFManager>())
{

}

Renderer::~Renderer() {
    cleanup();
}

void Renderer::resize(vk::raii::Device& device, HANDLE sharedMemoryHandle,
    vk::Extent2D extent, size_t allocationSize) {
    cleanup();
    m_extent = extent;

    cudaExternalMemoryHandleDesc extMemDesc{};
    extMemDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    extMemDesc.handle.win32.handle = sharedMemoryHandle;
    extMemDesc.size = allocationSize;

    if (cudaImportExternalMemory(&m_cudaExtMemory, &extMemDesc) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to import external Vulkan memory handle!");
    }

    cudaExternalMemoryMipmappedArrayDesc mipDesc{};
    mipDesc.offset = 0;
    mipDesc.formatDesc.x = 8; 
    mipDesc.formatDesc.y = 8;
    mipDesc.formatDesc.z = 8; 
    mipDesc.formatDesc.w = 8;
    mipDesc.formatDesc.f = cudaChannelFormatKindUnsigned;
    mipDesc.extent.width = m_extent.width;
    mipDesc.extent.height = m_extent.height;
    mipDesc.extent.depth = 0;
    mipDesc.numLevels = 1;

    if (cudaExternalMemoryGetMappedMipmappedArray(&m_cudaMipmappedArray, m_cudaExtMemory, &mipDesc) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to map external mipmapped array!");
    }

    if (cudaGetMipmappedArrayLevel(&m_cudaArray, m_cudaMipmappedArray, 0) != cudaSuccess) {
		throw std::runtime_error("CUDA Failed to get mipmapped array level");
	}

    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_cudaArray;

    if (cudaCreateSurfaceObject(&m_cudaSurfaceObject, &resDesc) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to create writable surface object pointer!");
    }

    if (cudaMalloc((void**)&dev_pathStates, getPixelCount() * sizeof(PathState)) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to allocate dev_pathStates");
    }

    if (cudaMalloc((void**)&dev_intersectionData, getPixelCount() * sizeof(IntersectionData)) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to allocate dev_intersectionData");
    }

    if (cudaMalloc((void**)&dev_sampleCounts, getPixelCount() * sizeof(unsigned int)) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to allocate dev_sampleCounts");
    }

    if (cudaMalloc((void**)&dev_accumulatedColor, getPixelCount() * sizeof(glm::vec3)) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to allocate dev_accumulatedColor");
    }

    cudaMemset(dev_sampleCounts, 0, getPixelCount() * sizeof(unsigned int));

    cudaMemset(dev_accumulatedColor, 0, getPixelCount() * sizeof(glm::vec3));

    if (m_svgfManager != nullptr) {
        m_svgfManager->resize(m_extent.width, m_extent.height);
    }
}

void Renderer::render(const std::unique_ptr<Scene>& scene, const std::unique_ptr<EnvironmentMap>& environmentMap, const Camera& camera, bool bClearAccumulatedSamples)
{
    if (m_cudaSurfaceObject == 0) {
        return;
    }

    if (bClearAccumulatedSamples) {
        cudaMemset(dev_sampleCounts, 0, getPixelCount() * sizeof(unsigned int));
        cudaMemset(dev_accumulatedColor, 0.0f, getPixelCount() * sizeof(glm::vec3));
        m_frameIndex = 0;
    }

    int initialActivePathCount = getPixelCount();
    int currentActivePathCount = initialActivePathCount;
    int maxBounces = 6;

    launchCameraRayGenKernel(
        dev_pathStates,
        m_extent.width,
        m_extent.height,
        camera,
        m_frameIndex);

    for (int i = 0; i < maxBounces; i++) {
        if (currentActivePathCount <= 0) {
            break;
        }

        launchIntersectKernel(dev_pathStates,
            dev_intersectionData,
            scene,
            currentActivePathCount);

        if (i == 0 && m_svgfManager != nullptr) {
            m_svgfManager->captureGBuffer(dev_pathStates, dev_intersectionData, currentActivePathCount, camera);
        }

        launchShadeKernel(dev_pathStates,
            dev_intersectionData,
            scene,
            environmentMap,
            currentActivePathCount,
            m_cudaSurfaceObject,
            dev_accumulatedColor,
            m_svgfManager != nullptr ? m_svgfManager->dev_pingBuffer : nullptr,
            dev_sampleCounts,
            m_extent.width,
            i,
            m_frameIndex);

        // Run stream compaction
        currentActivePathCount = runStreamCompaction(dev_pathStates, dev_intersectionData, currentActivePathCount);
    }

    if (currentActivePathCount > 0) {
        launchColorSurfaceKernel(dev_pathStates, 
            currentActivePathCount, 
            m_cudaSurfaceObject, 
            dev_accumulatedColor,
            m_svgfManager != nullptr ? m_svgfManager->dev_pingBuffer : nullptr,
            dev_sampleCounts, 
            m_extent.width);
    }

    if (m_svgfManager != nullptr) {
        m_svgfManager->executeTemporalAccumulation();

        m_svgfManager->executeVarianceEstimation();

        m_svgfManager->executeAtrousFilteringPipeline();

        //m_svgfManager->debugMotionVectors(m_cudaSurfaceObject);
        m_svgfManager->debugIlluminance(m_cudaSurfaceObject);
        //m_svgfManager->debugVariance(m_cudaSurfaceObject);

        m_svgfManager->swapBuffers();
    }

    m_frameIndex++;

    cudaDeviceSynchronize();
}

void Renderer::saveCurrentRenderToFile(const std::string& filepath)
{
    int numPixels = getPixelCount();
    if (numPixels <= 0) {
        return;
    }

    // Copy buffer of accumulated 
    std::vector<glm::vec3> cpuAccumulatedColor(numPixels);
    std::vector<unsigned int> cpuSampleCounts(numPixels);

    if (cudaMemcpy(cpuAccumulatedColor.data(), dev_accumulatedColor, numPixels * sizeof(glm::vec3), cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("CUDA failed to copy dev_accumulatedColor to the cpu");
    }

    if (cudaMemcpy(cpuSampleCounts.data(), dev_sampleCounts, numPixels * sizeof(unsigned int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("CUDA failed to copy dev_sampleCounts to the cpu");
    }

    // Convert to format readable by stb image
    std::vector<uint8_t> outputImage(numPixels * 3);

    for (int i = 0; i < numPixels; i++) {
        unsigned int samples = cpuSampleCounts[i];
        glm::vec3 accumulatedColor = cpuAccumulatedColor[i];

        glm::vec3 color = glm::vec3(0.0f);

        if (samples > 0) {
            color = accumulatedColor / static_cast<float>(samples);
        }

        // Gamma correct like in the path tracer
        float r = std::clamp(std::pow(color.x, 1.0f / 2.2f), 0.0f, 1.0f);
        float g = std::clamp(std::pow(color.y, 1.0f / 2.2f), 0.0f, 1.0f);
        float b = std::clamp(std::pow(color.z, 1.0f / 2.2f), 0.0f, 1.0f);

        // Write into output image buffer
        outputImage[i * 3 + 0] = static_cast<uint8_t>(r * 255.99f);
        outputImage[i * 3 + 1] = static_cast<uint8_t>(g * 255.99f);
        outputImage[i * 3 + 2] = static_cast<uint8_t>(b * 255.99f);
    }

    stbi_flip_vertically_on_write(false);

    stbi_write_png(filepath.c_str(), m_extent.width, m_extent.height, 3, outputImage.data(), m_extent.width * 3);
}

void Renderer::setSVGFEnabled(bool bEnabled)
{
    if (bEnabled) {
        if (m_svgfManager == nullptr) {
            m_svgfManager = std::make_unique<SVGFManager>();
            m_svgfManager->resize(m_extent.width, m_extent.height);
        }
    }
    else {
        m_svgfManager = nullptr;
    }
}

void Renderer::cleanup()
{
    if (m_cudaSurfaceObject) {
        cudaDestroySurfaceObject(m_cudaSurfaceObject);
        m_cudaSurfaceObject = 0;
    }
    if (m_cudaMipmappedArray) {
        cudaFreeMipmappedArray(m_cudaMipmappedArray);
        m_cudaMipmappedArray = nullptr;
    }
    if (m_cudaExtMemory) {
        cudaDestroyExternalMemory(m_cudaExtMemory);
        m_cudaExtMemory = nullptr;
    }
    m_cudaArray = nullptr;

    if (dev_pathStates) {
        cudaFree(dev_pathStates);
        dev_pathStates = nullptr;
    }

    if (dev_intersectionData) {
        cudaFree(dev_intersectionData);
        dev_intersectionData = nullptr;
    }

    if (dev_sampleCounts) {
        cudaFree(dev_sampleCounts);
        dev_sampleCounts = nullptr;
    }

    if (dev_accumulatedColor) {
        cudaFree(dev_accumulatedColor);
        dev_sampleCounts = nullptr;
    }

    m_activeRayCount = 0;
}
