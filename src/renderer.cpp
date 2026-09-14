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

    if (cudaMalloc((void**)&dev_currentRays, getPixelCount() * sizeof(Ray)) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to allocate dev_currentRays");
    }

    if (cudaMalloc((void**)&dev_nextRays, getPixelCount() * sizeof(Ray)) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to allocate dev_nextRays");
    }
}

void Renderer::render()
{
    if (m_cudaSurfaceObject == 0) {
        return;
    }
    glm::vec3 cameraPos(0.0f, 0.0f, 3.0f);
    glm::vec3 cameraLook(0.0f, 0.0f, -1.0f);
    glm::vec3 cameraRight(1.0f, 0.0f, 0.0f);
    glm::vec3 cameraUp(0.0f, 1.0f, 0.0f);
    float fovY = 45.f;

    launchCameraRayGenKernel(
        dev_currentRays,
        dev_pathStates,
        m_extent.width,
        m_extent.height,
        cameraPos, cameraLook, cameraRight, cameraUp,
        fovY);

    launchDebugRaysKernel(dev_currentRays, m_cudaSurfaceObject, m_extent.width, m_extent.height);
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

    if (dev_currentRays) {
        cudaFree(dev_currentRays);
        dev_currentRays = nullptr;
    }

    if (dev_nextRays) {
        cudaFree(dev_nextRays);
        dev_nextRays = nullptr;
    }

    m_activeRayCount = 0;
}
