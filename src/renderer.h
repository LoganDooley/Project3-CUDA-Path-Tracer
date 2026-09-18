#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include "pathTraceCommon.h"

class Renderer {
public:
	Renderer();
	~Renderer();

	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer&) = delete;

	void resize(vk::raii::Device& device, HANDLE sharedMemoryHandle,
		vk::Extent2D extent, size_t allocationSize);

	void render();

private:
	void cleanup();

	int getPixelCount() const { return m_extent.width * m_extent.height; }

	cudaExternalMemory_t m_cudaExtMemory = nullptr;
	cudaMipmappedArray_t m_cudaMipmappedArray = nullptr;
	cudaArray_t m_cudaArray = nullptr;
	cudaSurfaceObject_t m_cudaSurfaceObject = 0;

	vk::Extent2D m_extent;

	PathState* dev_pathStates = nullptr;

	int m_activeRayCount = 0;
};