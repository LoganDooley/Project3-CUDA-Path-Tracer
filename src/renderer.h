#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

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

	cudaExternalMemory_t m_cudaExtMemory = nullptr;
	cudaMipmappedArray_t m_cudaMipmappedArray = nullptr;
	cudaArray_t m_cudaArray = nullptr;
	cudaSurfaceObject_t m_cudaSurfaceObject = 0;

	vk::Extent2D m_extent;
};