#include "environmentMap.h"

#include <stdexcept>

EnvironmentMap::EnvironmentMap(float* imageData, size_t width, size_t height)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    if (cudaMallocArray(&m_environmentMapArray, &channelDesc, width, height) != cudaSuccess) {
        throw std::runtime_error("CUDA: Failed to malloc m_environmentMapArray");
        return;
    }

    size_t pitch = width * sizeof(float) * 4;
    cudaError_t memcpyResult = cudaMemcpy2DToArray(
        m_environmentMapArray,
        0, 0,
        imageData,
        pitch,
        pitch,
        height,
        cudaMemcpyHostToDevice
    );
    if (memcpyResult != cudaSuccess) {
        throw std::runtime_error("CUDA: Failed to copy image data to m_environmentMapArray");
        return;
    }

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = m_environmentMapArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeWrap;  // Wrap U for spherical map
    texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp V for poles
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    if (cudaCreateTextureObject(&m_environmentMapTexture, &resDesc, &texDesc, nullptr) != cudaSuccess) {
        cudaFreeArray(m_environmentMapArray);
        m_environmentMapArray = nullptr;
        m_environmentMapTexture = 0;
    }
}

EnvironmentMap::~EnvironmentMap()
{
    if (m_environmentMapTexture != 0) {
        cudaDestroyTextureObject(m_environmentMapTexture);
        m_environmentMapTexture = 0;
    }

    if (m_environmentMapArray != nullptr) {
        cudaFreeArray(m_environmentMapArray);
        m_environmentMapArray = nullptr;
    }
}

EnvironmentMap& EnvironmentMap::operator=(EnvironmentMap&& other) noexcept
{
    if (this != &other) {
        if (m_environmentMapTexture != 0) {
            cudaDestroyTextureObject(m_environmentMapTexture);
            m_environmentMapTexture = 0;
        }

        if (m_environmentMapArray != nullptr) {
            cudaFreeArray(m_environmentMapArray);
            m_environmentMapArray = nullptr;
        }

        m_environmentMapTexture = other.m_environmentMapTexture;
        m_environmentMapArray = other.m_environmentMapArray;

        other.m_environmentMapTexture = 0;
        other.m_environmentMapArray = nullptr;
    }

    return *this;
}

EnvironmentMap::EnvironmentMap(EnvironmentMap&& other) noexcept :
    m_environmentMapTexture(other.m_environmentMapTexture),
    m_environmentMapArray(other.m_environmentMapArray)
{
    other.m_environmentMapTexture = 0;
    other.m_environmentMapArray = nullptr;
}
