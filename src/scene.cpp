#include "scene.h"

// External Includes
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>

// STD Includes
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <unordered_map>

using json = nlohmann::json;

static glm::vec3 parseVec3(const json& arrayNode) {
    if (!arrayNode.is_array() || arrayNode.size() < 3) {
        return glm::vec3(0.0f);
    }
    return glm::vec3(arrayNode[0].get<float>(), arrayNode[1].get<float>(), arrayNode[2].get<float>());
}

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
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open scene file: " + filepath);
    }

    json sceneJson;
    try {
        file >> sceneJson;
    }
    catch (const json::parse_error& e) {
        throw std::runtime_error("Failed to parse scene file: " + std::string(e.what()));
    }

    std::vector<Material> materials;
    std::vector<Geom> geometry;
    std::unordered_map<std::string, int> materialMap;

    // Parse materials
    if (sceneJson.contains("Materials")) {
        for (const auto& [name, data] : sceneJson["Materials"].items()) {
            Material mat{};
            if (data.contains("RGB")) {
                mat.color = parseVec3(data["RGB"]);
            }

            std::string type = data.value("TYPE", "Diffuse");
            if (type == "Emitting") {
                mat.emittance = data.value("EMITTANCE", 0.0f);
            }
            else if (type == "Specular") {
                mat.hasReflective = 1.0f;
                float roughness = data.value("ROUGHNESS", 0.0f);
                mat.specular.exponent = glm::mix(1000.0f, 1.0f, roughness);
                mat.specular.color = mat.color;
            }

            materials.push_back(mat);
            materialMap[name] = static_cast<int>(materials.size()) - 1;
        }
    }
    if (materials.empty()) {
        // Add a fallback
        materials.push_back(Material{});
    }

    // Parse geometry
    if (sceneJson.contains("Objects")) {
        for (const auto& objData : sceneJson["Objects"]) {
            Geom geom{};

            std::string type = objData.value("TYPE", "cube");
            geom.type = (type == "sphere") ? GeomType::SPHERE : GeomType::CUBE;

            std::string matName = objData.value("MATERIAL", "");
            auto it = materialMap.find(matName);
            geom.materialid = (it != materialMap.end()) ? it->second : 0;

            // Extract temporary transform components
            glm::vec3 translation = parseVec3(objData["TRANS"]);
            glm::vec3 rotation = parseVec3(objData["ROTAT"]);
            glm::vec3 scale = parseVec3(objData["SCALE"]);

            // Convert to matrices
            glm::mat4 translationMat = glm::translate(glm::mat4(1.0f), translation);
            glm::mat4 rotationMat = glm::mat4(1.0f);
            rotationMat = glm::rotate(rotationMat, glm::radians(rotation.x), glm::vec3(1, 0, 0));
            rotationMat = glm::rotate(rotationMat, glm::radians(rotation.y), glm::vec3(0, 1, 0));
            rotationMat = glm::rotate(rotationMat, glm::radians(rotation.z), glm::vec3(0, 0, 1));
            glm::mat4 scaleMat = glm::scale(glm::mat4(1.0f), scale);

            // Set matrices
            geom.transform = translationMat * rotationMat * scaleMat;
            geom.inverseTransform = glm::inverse(geom.transform);
            geom.invTranspose = glm::inverseTranspose(geom.transform);

            geometry.push_back(geom);
        }
    }

	std::unique_ptr<Scene> scene = std::make_unique<Scene>();

    // Allocate geometry and material space on the GPU
    if (geometry.size() > 0) {
        if (cudaMalloc((void**)&scene->dev_geometry, geometry.size() * sizeof(Geom)) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to allocate dev_geometry");
        }
    }

    // We always have at least one fallback material present
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
