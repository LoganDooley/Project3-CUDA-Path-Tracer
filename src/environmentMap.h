#pragma once

#include <cuda_runtime.h>

#include <vector>

class EnvironmentMap {
public:
	EnvironmentMap(float* imageData, size_t width, size_t height);
	~EnvironmentMap();

	EnvironmentMap& operator=(const EnvironmentMap&) = delete;
	EnvironmentMap(const EnvironmentMap&) = delete;

	EnvironmentMap& operator=(EnvironmentMap&&) noexcept;
	EnvironmentMap(EnvironmentMap&&) noexcept;

	cudaTextureObject_t m_environmentMapTexture = 0;

private:
	cudaArray_t m_environmentMapArray = nullptr;
};