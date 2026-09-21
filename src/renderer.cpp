#include "renderer.h"

#include "kernel.h"

Renderer::Renderer() {

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
}

void Renderer::render(const std::unique_ptr<Scene>& scene, const Camera& camera, bool bClearAccumulatedSamples)
{
    if (m_cudaSurfaceObject == 0) {
        return;
    }

    if (bClearAccumulatedSamples) {
        cudaMemset(dev_sampleCounts, 0, getPixelCount() * sizeof(unsigned int));
        cudaMemset(dev_accumulatedColor, 0.0f, getPixelCount() * sizeof(glm::vec3));
    }

    int initialActivePathCount = getPixelCount();
    int currentActivePathCount = initialActivePathCount;
    int maxBounces = 3;

    launchCameraRayGenKernel(
        dev_pathStates,
        m_extent.width,
        m_extent.height,
        camera);

    for (int i = 0; i < maxBounces; i++) {
        if (currentActivePathCount <= 0) {
            break;
        }

        launchIntersectKernel(dev_pathStates,
            dev_intersectionData,
            scene ? scene->dev_geometry : nullptr,
            scene ? scene->m_geometryCount : 0,
            currentActivePathCount);

        launchShadeKernel(dev_pathStates,
            dev_intersectionData,
            scene ? scene->dev_materials : nullptr,
            scene ? scene->m_materialCount : 0,
            currentActivePathCount,
            m_cudaSurfaceObject,
            dev_accumulatedColor,
            dev_sampleCounts,
            m_extent.width,
            i);

        // Run stream compaction
        currentActivePathCount = runStreamCompaction(dev_pathStates, dev_intersectionData, currentActivePathCount);
    }

    if (currentActivePathCount > 0) {
        launchColorSurfaceKernel(dev_pathStates, 
            currentActivePathCount, 
            m_cudaSurfaceObject, 
            dev_accumulatedColor,
            dev_sampleCounts, 
            m_extent.width);
    }

    cudaDeviceSynchronize();
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
