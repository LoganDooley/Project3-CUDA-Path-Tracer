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

		bool success = CreateTextureFromPixels(data, width, height, textureObject, textureArray);
		stbi_image_free(data);
		return success;
	}

	// Binary gltf files have the textures in memory so have to use a special helper here
	static bool LoadTextureFromMemory(const unsigned char* encodedBytes, size_t byteCount, cudaTextureObject_t& textureObject, cudaArray_t& textureArray) {
		int width, height, channels;
		unsigned char* data = stbi_load_from_memory(encodedBytes, static_cast<int>(byteCount), &width, &height, &channels, 4);

		if (!data) {
			std::cerr << "Failed to decode embedded texture: " << stbi_failure_reason() << std::endl;
			return false;
		}

		bool success = CreateTextureFromPixels(data, width, height, textureObject, textureArray);
		stbi_image_free(data);
		return success;
	}

private:
	// Helper for taking pixel data and uploading it to the gpu
	static bool CreateTextureFromPixels(const unsigned char* data, int width, int height, cudaTextureObject_t& textureObject, cudaArray_t& textureArray) {
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		cudaError_t err = cudaMallocArray(&textureArray, &channelDesc, width, height);
		if (err != cudaSuccess) {
			std::cerr << "[CUDA] Failed to allocate array: " << cudaGetErrorString(err) << std::endl;
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
			textureArray = nullptr;
			return false;
		}

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
			textureArray = nullptr;
			return false;
		}

		return true;
	}
};
