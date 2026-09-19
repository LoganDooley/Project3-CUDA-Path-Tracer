#include "scene.h"

#include <cuda_runtime.h>

#include <stdexcept>

Scene::Scene()
{
}

Scene::~Scene()
{
    if (dev_geometry) {
        cudaFree(dev_geometry);
        dev_geometry = nullptr;
    }

    if (dev_materials) {
        cudaFree(dev_materials);
        dev_materials = nullptr;
    }
}

const Scene& Scene::operator=(Scene&& other) noexcept
{
    if (this != &other) {
        if (dev_geometry) {
            cudaFree(dev_geometry);
        }
        if (dev_materials) {
            cudaFree(dev_materials);
        }

        dev_geometry = other.dev_geometry;
        dev_materials = other.dev_materials;

        other.dev_geometry = nullptr;
        other.dev_materials = nullptr;
    }

    return *this;
}

Scene::Scene(Scene&& other) noexcept :
    dev_geometry(other.dev_geometry),
    dev_materials(other.dev_materials)
{
    other.dev_geometry = nullptr;
    other.dev_materials = nullptr;
}

std::unique_ptr<Scene> SceneLoader::loadFromFile(const std::string& filepath)
{
	std::unique_ptr<Scene> scene = std::make_unique<Scene>();

    // Parse geometry and materials on the CPU
    std::vector<Geom> geometry = {
        Geom{}
    };

    std::vector<Material> materials = {
        Material{}
    };

    // Allocate geometry and material space on the GPU
    if (cudaMalloc((void**)&scene->dev_geometry, geometry.size() * sizeof(Geom)) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to allocate dev_geometry");
    }

    if (cudaMalloc((void**)&scene->dev_materials, materials.size() * sizeof(Material)) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to allocate dev_materials");
    }

    // Move geometry and materials to the GPU
    if (cudaMemcpy(scene->dev_geometry, geometry.data(), geometry.size() * sizeof(Geom), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to memcopy geometry to dev_geometry");
    }

    if (cudaMemcpy(scene->dev_materials, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("CUDA Failed to memcopy materials to dev_materials");
    }

    scene->m_geometryCount = geometry.size();
    scene->m_materialCount = materials.size();

    return scene;
}
