#pragma once

#include "stb_image.h"
#include <iostream>
#include <cuda_runtime.h>

class Texture {
public:
	static bool LoadTexture(const char* filepath, cudaTextureObject_t& textureObject, cudaArray_t& textureArray) {
		int width, height, channels;
		unsigned char* data = stbi_load(filepath, &width, &height, &channels, 4);

		if(!data) {
			std::cerr << "Failed to load texture: " << filepath << std::endl;
			return false;
		}

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		cudaError_t err = cudaMallocArray(&textureArray, &channelDesc, width, height);
		if (err != cudaSuccess) {
			std::cerr << "[CUDA] Failed to allocate array: " << cudaGetErrorString(err) << std::endl;
			stbi_image_free(data);
			return false;
		}

		size_t pitch = width * 4 * sizeof(unsigned char);
		cudaError_t memcpyResult = cudaMemcpy2DToArray(
			textureArray,
			0, 0,
			data,
			pitch,
			pitch,
			height,
			cudaMemcpyHostToDevice
		);
		if(memcpyResult != cudaSuccess) {
			std::cerr << "[CUDA] Failed to copy texture data to array: " << cudaGetErrorString(memcpyResult) << std::endl;
			cudaFreeArray(textureArray);
			stbi_image_free(data);
			return false;
		}

		stbi_image_free(data);

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = textureArray;

		cudaTextureDesc texDesc = {};
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;

		texDesc.readMode = cudaReadModeNormalizedFloat;
		texDesc.normalizedCoords = 1;

		err = cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr);
		if(err != cudaSuccess) {
			std::cerr << "[CUDA] Failed to create texture object: " << cudaGetErrorString(err) << std::endl;
			cudaFreeArray(textureArray);
			return false;
		}

		return true;
	}
};