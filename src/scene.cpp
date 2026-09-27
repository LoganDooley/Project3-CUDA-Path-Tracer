#include "scene.h"

// External Includes
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include "tiny_gltf_v3.h"

// STD Includes
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <iterator>

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

Scene& Scene::operator=(Scene&& other) noexcept
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
	std::filesystem::path path(filepath);
	std::string ext = path.extension().string();

    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if(ext == ".json") {
        return loadFromJson(filepath);
    }
    else if(ext == ".gltf" || ext == ".glb") {
        return loadFromGltf(filepath);
    }
    else {
		std::cerr << "Unsupported scene file format: " << ext << std::endl;
        return nullptr;
	}
}

std::unique_ptr<Scene> SceneLoader::loadFromJson(const std::string& filepath)
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
                float roughness = data.value("ROUGHNESS", 0.0f);
                mat.specular.exponent = glm::mix(1000.0f, 1.0f, roughness);
                mat.specular.color = mat.color;
            }
            else if (type == "Mirror") {
                mat.hasReflective = 1.0f;
            }
            else if (type == "Glass") {
                mat.hasRefractive = 1.0f;
                mat.indexOfRefraction = data.value("IOR", 1.5f);
                mat.hasReflective = 1.0f;
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

    // Sort geometry so lights are in the front
    auto it = std::partition(geometry.begin(), geometry.end(), [materials](const Geom& geometry) {
        return geometry.materialid < materials.size() && geometry.materialid >= 0 && materials[geometry.materialid].emittance > 0.0f;
        });

    scene->m_lightCount = std::distance(geometry.begin(), it);

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

void parseGltfNodeRecursive(
    const tg3_model& model, 
    uint32_t nodeIdx, 
    const glm::mat4& parentTransform, 
    std::vector<Geom>& outGeometries, 
    std::vector<Triangle>& outTriangles)
{
    const tg3_node& node = model.nodes[nodeIdx];

    // Get the local transform of the node
    glm::mat4 localTransform = glm::mat4(1.0f);
    if (node.has_matrix) {
        localTransform = glm::make_mat4(node.matrix);
    }
    else {
        glm::vec3 translation = glm::make_vec3(node.translation);
        glm::quat rotation = glm::make_quat(node.rotation);
        glm::vec3 scale = glm::make_vec3(node.scale);

        localTransform = glm::translate(glm::mat4(1.0f), translation) * glm::mat4_cast(rotation) * glm::scale(glm::mat4(1.0f), scale);
    }

    // Use parent transform to get global transform
    glm::mat4 globalTransform = parentTransform * localTransform;

	// If the node has a mesh, process it
    if (node.mesh >= 0 && node.mesh < model.meshes_count) {
        const tg3_mesh& mesh = model.meshes[node.mesh];

        // Each primitive will be a geometry struct
        for (uint32_t i = 0; i < mesh.primitives_count; i++) {
            const tg3_primitive& prim = mesh.primitives[i];

            // Get position accessor
            int positionAccessorIdx = -1;
            for (uint32_t a = 0; a < prim.attributes_count; a++) {
                if (strcmp(prim.attributes[a].key.data, "POSITION") == 0) {
                    positionAccessorIdx = prim.attributes[a].value;
                    break;
                }
            }

            if (positionAccessorIdx < 0 || positionAccessorIdx >= model.accessors_count) {
                continue;
            }

            // Get vertices in local space
            const tg3_accessor& posAccessor = model.accessors[positionAccessorIdx];
            const tg3_buffer_view& posView = model.buffer_views[posAccessor.buffer_view];
            const tg3_buffer& posBuffer = model.buffers[posView.buffer];

            const uint8_t* posBufferData = posBuffer.data.data + posView.byte_offset + posAccessor.byte_offset;
            uint32_t posStride = posView.byte_stride > 0 ? posView.byte_stride : sizeof(float) * 3;
            uint32_t vertexCount = posAccessor.count;

            // Extract raw local vertices into a temporary vector
            std::vector<glm::vec3> localVertices(vertexCount);
            for (uint32_t v = 0; v < vertexCount; v++) {
                const float* rawVertexFloats = reinterpret_cast<const float*>(posBufferData + (v * posStride));

                localVertices[v] = glm::vec3(rawVertexFloats[0], rawVertexFloats[1], rawVertexFloats[2]);
            }

            // Assemble triangles
            int startTriangleOffset = static_cast<int>(outTriangles.size());
            int primitiveTriangleCount = 0;

            if (prim.indices >= 0 && static_cast<uint32_t>(prim.indices) < model.accessors_count) {
                // Mesh uses indexing

				// Get index accessor
                const tg3_accessor& idxAccessor = model.accessors[prim.indices];
                const tg3_buffer_view& idxView = model.buffer_views[idxAccessor.buffer_view];
                const tg3_buffer& idxBuffer = model.buffers[idxView.buffer];

                const uint8_t* idxBufferData = idxBuffer.data.data + idxView.byte_offset + idxAccessor.byte_offset;
                uint32_t idxStride = idxView.byte_stride > 0 ? idxView.byte_stride : (idxAccessor.component_type == 5123 ? 2 : 4); // 5123 is GL_UNSIGNED_SHORT

                uint32_t indexCount = idxAccessor.count;
				primitiveTriangleCount = indexCount / 3;

                for (uint32_t idx = 0; idx < indexCount; idx += 3) {
                    uint32_t i0 = 0, i1 = 0, i2 = 0;

                    if (idxAccessor.component_type == 5123) { // GL_UNSIGNED_SHORT
                        const uint16_t* ptr = reinterpret_cast<const uint16_t*>(idxBufferData + (idx * idxStride));
                        i0 = ptr[0]; i1 = ptr[1]; i2 = ptr[2];
                    }
                    else { // GL_UNSIGNED_INT
                        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(idxBufferData + (idx * idxStride));
                        i0 = ptr[0]; i1 = ptr[1]; i2 = ptr[2];
                    }

					Triangle tri{};
					tri.v0 = localVertices[i0];
					tri.v1 = localVertices[i1];
					tri.v2 = localVertices[i2];
                    outTriangles.push_back(tri);
                }
            }
            else {
                // No indexing, just every 3 vertices are a triangle
				primitiveTriangleCount = vertexCount / 3;
                for (uint32_t v = 0; v < vertexCount; v += 3) {
                    Triangle tri{};
					tri.v0 = localVertices[v + 0];
					tri.v1 = localVertices[v + 1];
					tri.v2 = localVertices[v + 2];
					outTriangles.push_back(tri);
                }
            }

            // Get surface area (bad method for MIS temporarily)
			float surfaceArea = 0.0f;
            for(int t = startTriangleOffset; t < startTriangleOffset + primitiveTriangleCount; t++) {
                const Triangle& tri = outTriangles[t];
                glm::vec3 edge1 = tri.v1 - tri.v0;
                glm::vec3 edge2 = tri.v2 - tri.v0;
                surfaceArea += 0.5f * glm::length(glm::cross(edge1, edge2));
			}

            Geom geom{};
            geom.type = GeomType::MESH;

            // Grab material index (fallback to material 0)
            geom.materialid = (prim.material >= 0 && prim.material < model.materials_count) ? static_cast<int>(prim.material) : 0;

            geom.transform = globalTransform;
            geom.inverseTransform = glm::inverse(geom.transform);
            geom.invTranspose = glm::inverseTranspose(geom.transform);

			geom.triangleOffset = startTriangleOffset;
            geom.triangleCount = primitiveTriangleCount;
			geom.surfaceArea = surfaceArea;

            outGeometries.push_back(geom);
        }
    }

    // Recurse to children
    for (uint32_t i = 0; i < node.children_count; i++) {
        parseGltfNodeRecursive(model, node.children[i], globalTransform, outGeometries, outTriangles);
    }
}

void parseGltfMaterials(const tg3_model& model, std::vector<Material>& outMaterials)
{
    for (uint32_t i = 0; i < model.materials_count; ++i) {
        const tg3_material& gltfMat = model.materials[i];
        Material mat{};

        float emissiveSum = gltfMat.emissive_factor[0] + gltfMat.emissive_factor[1] + gltfMat.emissive_factor[2];
        
        if (emissiveSum > 0.0f) {
            mat.color = glm::vec3(
                gltfMat.emissive_factor[0],
                gltfMat.emissive_factor[1],
                gltfMat.emissive_factor[2]
			);
            mat.emittance = 1.0f;
        }
        else {
            mat.color = glm::vec3(
                gltfMat.pbr_metallic_roughness.base_color_factor[0],
                gltfMat.pbr_metallic_roughness.base_color_factor[1],
                gltfMat.pbr_metallic_roughness.base_color_factor[2]
			);
            mat.emittance = 0.0f;
        }

        float roughness = gltfMat.pbr_metallic_roughness.roughness_factor;
        if (roughness < 0.08f) {
            mat.hasReflective = 1.0f;
        }
        else {
            mat.specular.exponent = glm::mix(1000.0f, 1.0f, roughness);
            mat.specular.color = mat.color;
        }

        outMaterials.push_back(mat);
    }

    if (outMaterials.empty()) {
        // Add a fallback material if none are present
        outMaterials.push_back(Material{});
	}
}

std::unique_ptr<Scene> SceneLoader::loadFromGltf(const std::string& filepath)
{
    tg3_parse_options opts;
    tg3_error_stack errors;
    tg3_model model;

    tg3_parse_options_init(&opts);
    tg3_error_stack_init(&errors);

    tg3_error_code err = tg3_parse_file(&model, &errors, filepath.c_str(), static_cast<uint32_t>(filepath.length()), &opts);

    if (err != TG3_OK) {
        std::string errorsSummary = "";
        for (uint32_t i = 0; i < errors.count; i++) {
            errorsSummary += "\n[TinyGLTF3 Error] " + std::string(errors.entries[i].message);
        }
        tg3_error_stack_free(&errors);
		std::cerr << "Failed to parse glTF file: " << filepath << errorsSummary << std::endl;
        return nullptr;
    }

	std::cout << "Finished parsing glTF file: " << filepath << std::endl;

    std::vector<Material> materials;
	std::vector<Geom> geometry;
	std::vector<Triangle> triangles;

	parseGltfMaterials(model, materials);

	std::cout << " Finished parsing materials: " << materials.size() << std::endl;

    // Parse geometry from default scene
	uint32_t activeSceneIdx = (model.default_scene >= 0 && model.default_scene < model.scenes_count) ? model.default_scene : 0;
    if(activeSceneIdx < model.scenes_count) {
        const tg3_scene& scene = model.scenes[activeSceneIdx];
        for (uint32_t i = 0; i < scene.nodes_count; ++i) {
            parseGltfNodeRecursive(model, scene.nodes[i], glm::mat4(1.0f), geometry, triangles);
        }
	}

	std::cout << " Finished parsing geometry: " << geometry.size() << " and triangles: " << triangles.size() << std::endl;

    // Done with parsing
    tg3_model_free(&model);
    tg3_error_stack_free(&errors);

	std::unique_ptr<Scene> scene = std::make_unique<Scene>();

    // Sort geometry so lights are first
    auto it = std::partition(geometry.begin(), geometry.end(), [materials](const Geom& geom) {
        return geom.materialid < materials.size() && geom.materialid >= 0 && materials[geom.materialid].emittance > 0.0f;
        });
    scene->m_lightCount = std::distance(geometry.begin(), it);

	std::cout << " Finished sorting geometry, light count: " << scene->m_lightCount << std::endl;

    // Allocate geometry
    if (!geometry.empty()) {
        if (cudaMalloc((void**)&scene->dev_geometry, geometry.size() * sizeof(Geom)) != cudaSuccess) {

            throw std::runtime_error("CUDA Failed to allocate dev_geometry");
		}
        if (cudaMemcpy(scene->dev_geometry, geometry.data(), geometry.size() * sizeof(Geom), cudaMemcpyHostToDevice) != cudaSuccess) {
			throw std::runtime_error("CUDA Failed to memcopy geometry to dev_geometry");
        }
    }

	std::cout << " Finished allocating geometry" << std::endl;

    // Allocate materials
    if(!materials.empty()) {
        if (cudaMalloc((void**)&scene->dev_materials, materials.size() * sizeof(Material)) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to allocate dev_materials");
        }
        if (cudaMemcpy(scene->dev_materials, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to memcopy materials to dev_materials");
		}
	}

	std::cout << " Finished allocating materials" << std::endl;

    // Allocate triangles
    if (!triangles.empty()) {
        if (cudaMalloc((void**)&scene->dev_triangles, triangles.size() * sizeof(Triangle)) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to allocate dev_triangles");
        }
        if(cudaMemcpy(scene->dev_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to memcopy triangles to dev_triangles");
		}
    }

	std::cout << " Finished allocating triangles" << std::endl;

    scene->m_geometryCount = geometry.size();
	scene->m_materialCount = materials.size();
	scene->m_triangleCount = triangles.size();

    return scene;
}
