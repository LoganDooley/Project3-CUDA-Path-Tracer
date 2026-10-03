#include "scene.h"

#include "bvh.h"
#include "material.h"
#include "pathTraceCommon.h"
#include "texture.h"

// External Includes
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include "tiny_gltf_v3.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

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

Scene::Scene(const std::vector<Geom>& geometry,
    int lightCount,
    const std::vector<Triangle>& triangles, 
    const std::vector<BLASNode>& blasNodes, 
    const std::vector<TLASNode>& tlasNodes, 
    const std::vector<Material>& materials, 
    const std::vector<cudaTextureObject_t>& textures, 
    const std::vector<cudaArray_t>& textureArrays) :
	m_geometryCount(geometry.size()),
	m_lightCount(lightCount),
	m_triangleCount(triangles.size()),
	m_blasNodeCount(blasNodes.size()),
	m_tlasNodeCount(tlasNodes.size()),
	m_materialCount(materials.size()),
	m_textures(textures),
	m_textureArrays(textureArrays)
{
    if (!geometry.empty()) {
        if (cudaMalloc((void**)&dev_geometry, geometry.size() * sizeof(Geom)) != cudaSuccess) {

            throw std::runtime_error("CUDA Failed to allocate dev_geometry");
        }
        if (cudaMemcpy(dev_geometry, geometry.data(), geometry.size() * sizeof(Geom), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to memcopy geometry to dev_geometry");
        }
    }

    // Allocate materials
    if (!materials.empty()) {
        if (cudaMalloc((void**)&dev_materials, materials.size() * sizeof(Material)) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to allocate dev_materials");
        }
        if (cudaMemcpy(dev_materials, materials.data(), materials.size() * sizeof(Material), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to memcopy materials to dev_materials");
        }
    }

    // Allocate triangles
    if (!triangles.empty()) {
        if (cudaMalloc((void**)&dev_triangles, triangles.size() * sizeof(Triangle)) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to allocate dev_triangles");
        }
        if (cudaMemcpy(dev_triangles, triangles.data(), triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to memcopy triangles to dev_triangles");
        }
    }

    // Allocate BLAS nodes
    if (!blasNodes.empty()) {
        if (cudaMalloc((void**)&dev_blasNodes, blasNodes.size() * sizeof(BLASNode)) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to allocate dev_blasNodes");
        }
        if (cudaMemcpy(dev_blasNodes, blasNodes.data(), blasNodes.size() * sizeof(BLASNode), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to memcopy BLAS nodes to dev_blasNodes");
        }
    }

    // Allocate TLAS nodes
    if (!tlasNodes.empty()) {
        if (cudaMalloc((void**)&dev_tlasNodes, tlasNodes.size() * sizeof(TLASNode)) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to allocate dev_tlasNodes");
        }
        if (cudaMemcpy(dev_tlasNodes, tlasNodes.data(), tlasNodes.size() * sizeof(TLASNode), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("CUDA Failed to memcopy TLAS nodes to dev_tlasNodes");
        }
    }
}

Scene::~Scene()
{
    if (dev_geometry) {
        cudaFree(dev_geometry);
        dev_geometry = nullptr;
    }

    if (dev_triangles) {
        cudaFree(dev_triangles);
        dev_triangles = nullptr;
    }

    if (dev_blasNodes) {
        cudaFree(dev_blasNodes);
        dev_blasNodes = nullptr;
    }

    if (dev_tlasNodes) {
        cudaFree(dev_tlasNodes);
        dev_tlasNodes = nullptr;
    }

    if (dev_materials) {
        cudaFree(dev_materials);
        dev_materials = nullptr;
    }

    for(auto texture : m_textures) {
        if(texture) {
            cudaDestroyTextureObject(texture);
        }
	}
    m_textures.clear();

    for(auto textureArray : m_textureArrays) {
        if (textureArray) {
            cudaFreeArray(textureArray);
        }
    }
	m_textureArrays.clear();
}

Scene& Scene::operator=(Scene&& other) noexcept
{
    if (this != &other) {
		freeDeviceMemory();

        // Move from other scene
        dev_geometry = other.dev_geometry;
        m_geometryCount = other.m_geometryCount;
        dev_materials = other.dev_materials;
        m_materialCount = other.m_materialCount;
        dev_triangles = other.dev_triangles;
        m_triangleCount = other.m_triangleCount;
		dev_blasNodes = other.dev_blasNodes;
		m_blasNodeCount = other.m_blasNodeCount;
        dev_tlasNodes = other.dev_tlasNodes;
		m_tlasNodeCount = other.m_tlasNodeCount;
		m_textures = std::move(other.m_textures);
		m_textureArrays = std::move(other.m_textureArrays);
		
        // Clear other scene
        other.dev_geometry = nullptr;
        other.m_geometryCount = 0;
        other.dev_materials = nullptr;
        other.m_materialCount = 0;
		other.dev_triangles = nullptr;
		other.m_triangleCount = 0;
        other.dev_blasNodes = nullptr;
        other.m_blasNodeCount = 0;
		other.dev_tlasNodes = nullptr;
		other.m_tlasNodeCount = 0;
    }

    return *this;
}

Scene::Scene(Scene&& other) noexcept :
    dev_geometry(other.dev_geometry),
    m_geometryCount(other.m_geometryCount),
    dev_materials(other.dev_materials),
    m_materialCount(other.m_materialCount),
    dev_triangles(other.dev_triangles),
    m_triangleCount(other.m_triangleCount),
    dev_blasNodes(other.dev_blasNodes),
    m_blasNodeCount(other.m_blasNodeCount),
	dev_tlasNodes(other.dev_tlasNodes),
	m_tlasNodeCount(other.m_tlasNodeCount),
    m_textures(std::move(other.m_textures)),
	m_textureArrays(std::move(other.m_textureArrays))
{
    other.dev_geometry = nullptr;
    other.m_geometryCount = 0;
    other.dev_materials = nullptr;
    other.m_materialCount = 0;
    other.dev_triangles = nullptr;
    other.m_triangleCount = 0;
    other.dev_blasNodes = nullptr;
    other.m_blasNodeCount = 0;
    other.dev_tlasNodes = nullptr;
    other.m_tlasNodeCount = 0;
}

void Scene::freeDeviceMemory()
{
    if (dev_geometry) {
        cudaFree(dev_geometry);
        dev_geometry = nullptr;
    }

    if (dev_materials) {
        cudaFree(dev_materials);
        dev_materials = nullptr;
    }

    if (dev_triangles) {
        cudaFree(dev_triangles);
        dev_triangles = nullptr;
    }

    if (dev_blasNodes) {
        cudaFree(dev_blasNodes);
        dev_blasNodes = nullptr;
    }

    for (auto texture : m_textures) {
        if (texture) {
            cudaDestroyTextureObject(texture);
        }
    }
    m_textures.clear();

    for (auto textureArray : m_textureArrays) {
        if (textureArray) {
            cudaFreeArray(textureArray);
        }
    }
    m_textureArrays.clear();
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
    else if(ext == ".obj") {
        return loadFromObj(filepath);
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
                mat.albedo = parseVec3(data["RGB"]);
            }
            else {
				mat.albedo = glm::vec3(1.0f);
            }

            std::string type = data.value("TYPE", "Diffuse");
            if (type == "Emitting") {
                mat.emittance = data.value("EMITTANCE", 0.0f);
            }
            else if (type == "Specular") {
				mat.type = MaterialType::BlinnPhong;
                float roughness = data.value("ROUGHNESS", 0.0f);
                if(data.contains("SPECULAR_COLOR")) {
                    mat.blinnPhong.specularColor = parseVec3(data["SPECULAR_COLOR"]);
				}
                else {
					mat.blinnPhong.specularColor = glm::vec3(1.0f);
                }
				mat.blinnPhong.exponent = glm::mix(1.0f, 1000.0f, 1.0f - roughness);
				mat.blinnPhong.bRefractive = false;
            }
            else if (type == "Mirror") {
                mat.type = MaterialType::PerfectSpecular;
                mat.blinnPhong.bRefractive = false;
            }
            else if (type == "Glass") {
				mat.type = MaterialType::PerfectSpecular;
                mat.blinnPhong.bRefractive = true;
                mat.ior = data.value("IOR", 1.5f);
            }

            materials.push_back(mat);
            materialMap[name] = static_cast<int>(materials.size()) - 1;
        }
    }
    if (materials.empty()) {
        // Add a fallback diffuse pink material
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

    // Sort geometry so lights are first BEFORE building TLAS
    auto it = std::partition(geometry.begin(), geometry.end(), [materials](const Geom& geom) {
        return geom.materialid < materials.size() && geom.materialid >= 0 && materials[geom.materialid].emittance > 0.0f;
        });
    size_t lightCount = std::distance(geometry.begin(), it);

    // Build TLAS
    std::vector<TLASNode> tlasNodes;

    std::vector<GeometryBVHBuildData> geometryBuildData(geometry.size());
    for (size_t g = 0; g < geometry.size(); g++) {
        Geom& geom = geometry[g];
        AABB bounds = AABB::fromGeometry(geom, {});
        geometryBuildData[g].bounds = bounds;
        geometryBuildData[g].centroid = (bounds.min + bounds.max) * 0.5f;
        geometryBuildData[g].geometryIndex = g;
    }

    BVHBuilder::buildTLAS(geometryBuildData, 0, geometryBuildData.size(), tlasNodes);

    std::vector<Triangle> triangles;
	std::vector<BLASNode> blasNodes;

    return std::make_unique<Scene>(geometry, lightCount, triangles, blasNodes, tlasNodes, materials);
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
            int uvAccessorIdx = -1;
            for (uint32_t a = 0; a < prim.attributes_count; a++) {
                if (strcmp(prim.attributes[a].key.data, "POSITION") == 0) {
                    positionAccessorIdx = prim.attributes[a].value;
                }
                else if (strcmp(prim.attributes[a].key.data, "TEXCOORD_0") == 0) {
                    uvAccessorIdx = prim.attributes[a].value;
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

			const uint8_t* uvBufferData = nullptr;
			uint32_t uvStride = 0;
			bool hasUVs = false;

            if (uvAccessorIdx >= 0 && uvAccessorIdx < model.accessors_count) {
                const tg3_accessor& uvAccessor = model.accessors[uvAccessorIdx];
                const tg3_buffer_view& uvView = model.buffer_views[uvAccessor.buffer_view];
                const tg3_buffer& uvBuffer = model.buffers[uvView.buffer];
                
                uvBufferData = uvBuffer.data.data + uvView.byte_offset + uvAccessor.byte_offset;
				uvStride = uvView.byte_stride > 0 ? uvView.byte_stride : sizeof(float) * 2;
				hasUVs = uvAccessor.count == vertexCount;
            }

            // Extract raw local vertices into a temporary vector
            std::vector<glm::vec3> localVertices(vertexCount);
			std::vector<glm::vec2> localUVs(vertexCount, glm::vec2(0.0f));

            for (uint32_t v = 0; v < vertexCount; v++) {
                const float* rawVertexFloats = reinterpret_cast<const float*>(posBufferData + (v * posStride));
                localVertices[v] = glm::vec3(rawVertexFloats[0], rawVertexFloats[1], rawVertexFloats[2]);

                if(hasUVs) {
                    const float* rawUVFloats = reinterpret_cast<const float*>(uvBufferData + (v * uvStride));
                    localUVs[v] = glm::vec2(rawUVFloats[0], rawUVFloats[1]);
				}
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
					tri.uv0 = localUVs[i0];
					tri.uv1 = localUVs[i1];
					tri.uv2 = localUVs[i2];
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
					tri.uv0 = localUVs[v + 0];
					tri.uv1 = localUVs[v + 1];
					tri.uv2 = localUVs[v + 2];
					outTriangles.push_back(tri);
                }
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

            outGeometries.push_back(geom);
        }
    }

    // Recurse to children
    for (uint32_t i = 0; i < node.children_count; i++) {
        parseGltfNodeRecursive(model, node.children[i], globalTransform, outGeometries, outTriangles);
    }
}

void parseGltfMaterials(const tg3_model& model, const std::string& baseDir, std::vector<Material>& outMaterials, std::vector<cudaTextureObject_t>& outTextures, std::vector<cudaArray_t>& outTextureArrays)
{
    // Load all images as textures first
    if (model.images_count > 0) {
        outTextures.resize(model.images_count, 0);
		outTextureArrays.resize(model.images_count, nullptr);

        for (uint32_t i = 0; i < model.images_count; i++) {
            if(model.images[i].uri.data == nullptr) {
                std::cerr << "Image " << i << " has no URI, skipping." << std::endl;
                continue;
			}
            std::string fullImagePath = baseDir + model.images[i].uri.data;
            
            bool success = Texture::LoadTexture(
                fullImagePath.c_str(),
                outTextures[i],
                outTextureArrays[i]
            );

            if(!success) {
                std::cerr << "Failed to load texture: " << fullImagePath << std::endl;
			}
        }
    }

    // Parse materials and assign texture handles
    for (uint32_t i = 0; i < model.materials_count; ++i) {
        const tg3_material& gltfMat = model.materials[i];
        Material mat{};

        mat.type = MaterialType::PbrMetallicRoughness;
        mat.albedoTexture = 0;
		mat.pbr.metallicRoughnessTexture = 0;

        float emissiveSum = gltfMat.emissive_factor[0] + gltfMat.emissive_factor[1] + gltfMat.emissive_factor[2];
        if (emissiveSum > 0.0f) {
            mat.albedo = glm::vec3(
                gltfMat.emissive_factor[0],
                gltfMat.emissive_factor[1],
                gltfMat.emissive_factor[2]
			);
            mat.emittance = 1.0f;
            outMaterials.push_back(mat);
            continue;
        }

        mat.albedo = glm::vec3(
            gltfMat.pbr_metallic_roughness.base_color_factor[0],
            gltfMat.pbr_metallic_roughness.base_color_factor[1],
            gltfMat.pbr_metallic_roughness.base_color_factor[2]
        );
        mat.emittance = 0.0f;
		mat.pbr.roughness = gltfMat.pbr_metallic_roughness.roughness_factor;
		mat.pbr.metallic = gltfMat.pbr_metallic_roughness.metallic_factor;

        if (mat.pbr.roughness > 0.99f && mat.pbr.metallic == 0.0f && mat.pbr.transmission == 0.0f) {
            mat.type = MaterialType::OpaqueDiffuse;
        }

        if (gltfMat.pbr_metallic_roughness.base_color_texture.index >= 0 && 
            gltfMat.pbr_metallic_roughness.base_color_texture.index < model.textures_count) {
			int texIdx = gltfMat.pbr_metallic_roughness.base_color_texture.index;
            const tg3_texture& textureRef = model.textures[texIdx];

            if(textureRef.source >= 0 && textureRef.source < outTextures.size()) {
                mat.albedoTexture = outTextures[textureRef.source];
            }
            else {
                std::cerr << "Material " << i << " has invalid base color texture index: " << textureRef.source << std::endl;
			}
        }

        if(gltfMat.pbr_metallic_roughness.metallic_roughness_texture.index >= 0 && 
            gltfMat.pbr_metallic_roughness.metallic_roughness_texture.index < model.textures_count) {
            int texIdx = gltfMat.pbr_metallic_roughness.metallic_roughness_texture.index;
            const tg3_texture& textureRef = model.textures[texIdx];

            if (textureRef.source >= 0 && textureRef.source < outTextures.size()) {
                mat.pbr.metallicRoughnessTexture = outTextures[textureRef.source];
            }
            else {
				std::cerr << "Material " << i << " has invalid metallic-roughness texture index: " << textureRef.source << std::endl;
            }
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
    std::unique_ptr<Scene> scene = std::make_unique<Scene>();

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

    std::vector<Material> materials;
	std::vector<Geom> geometry;
	std::vector<Triangle> triangles;
	std::vector<cudaTextureObject_t> textures;
	std::vector<cudaArray_t> textureArrays;

    std::string baseDir = "";
    size_t lastSlash = filepath.find_last_of("/\\");
    if (lastSlash != std::string::npos) {
        baseDir = filepath.substr(0, lastSlash + 1);
    }
	parseGltfMaterials(model, baseDir, materials, textures, textureArrays);

    // Parse geometry from default scene

    constexpr float kImportScale = 1.0f;
    glm::mat4 rootTransform = glm::scale(glm::mat4(1.0f), glm::vec3(kImportScale));

	uint32_t activeSceneIdx = (model.default_scene >= 0 && model.default_scene < model.scenes_count) ? model.default_scene : 0;
    if(activeSceneIdx < model.scenes_count) {
        const tg3_scene& scene = model.scenes[activeSceneIdx];
        for (uint32_t i = 0; i < scene.nodes_count; ++i) {
            parseGltfNodeRecursive(model, scene.nodes[i], rootTransform, geometry, triangles);
        }
	}

    // Done with parsing
    tg3_model_free(&model);
    tg3_error_stack_free(&errors);

    // Build BLAS
    std::vector<BLASNode> blasNodes;
    std::vector<Triangle> sortedTriangles;

    for (size_t g = 0; g < geometry.size(); g++) {
        Geom& geom = geometry[g];

        if (geom.type != GeomType::MESH || geom.triangleCount <= 0) {
            continue;
        }

        std::vector<TriangleBVHBuildData> triangleBuildData(geom.triangleCount);
        for(int t = 0; t < geom.triangleCount; t++) {
            const Triangle& tri = triangles[geom.triangleOffset + t];
            AABB bounds = AABB::fromTriangle(tri);
			triangleBuildData[t].bounds = bounds;
			triangleBuildData[t].centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
            triangleBuildData[t].triangleIndex = geom.triangleOffset + t;
		}

		std::vector<BLASNode> geomBlasNodes;
        BVHBuilder::buildBLAS(triangleBuildData, 0, triangleBuildData.size(), geomBlasNodes, triangles, sortedTriangles);

        int nodeOffsetStart = blasNodes.size();
        for(size_t n = 0; n < geomBlasNodes.size(); n++) {
            BLASNode& node = geomBlasNodes[n];
            if(node.triangleCount == 0) {
                node.leftChild += nodeOffsetStart;
            }
		}

        geom.blasNodeOffset = nodeOffsetStart;

		blasNodes.insert(blasNodes.end(), geomBlasNodes.begin(), geomBlasNodes.end());
    }

    // Sort geometry so lights are first BEFORE building TLAS
    auto it = std::partition(geometry.begin(), geometry.end(), [materials](const Geom& geom) {
        return geom.materialid < materials.size() && geom.materialid >= 0 && materials[geom.materialid].emittance > 0.0f;
        });
    int lightCount = std::distance(geometry.begin(), it);

    // Build TLAS
    std::vector<TLASNode> tlasNodes;

    std::vector<GeometryBVHBuildData> geometryBuildData(geometry.size());
    for (size_t g = 0; g < geometry.size(); g++) {
        Geom& geom = geometry[g];
        AABB bounds = AABB::fromGeometry(geom, blasNodes);
        geometryBuildData[g].bounds = bounds;
        geometryBuildData[g].centroid = (bounds.min + bounds.max) * 0.5f;
        geometryBuildData[g].geometryIndex = g;
    }

    BVHBuilder::buildTLAS(geometryBuildData, 0, geometryBuildData.size(), tlasNodes);

    return std::make_unique<Scene>(geometry, lightCount, sortedTriangles, blasNodes, tlasNodes, materials, textures, textureArrays);
}

std::unique_ptr<Scene> SceneLoader::loadFromObj(const std::string& filepath)
{
	tinyobj::ObjReaderConfig readerConfig;

    std::filesystem::path path(filepath);
    std::string baseDir = path.parent_path().string();
    if (!baseDir.empty()) {
        baseDir += "/";
    }

	readerConfig.mtl_search_path = path.parent_path().string();
    readerConfig.triangulate = true;

	tinyobj::ObjReader reader;
    if(!reader.ParseFromFile(filepath, readerConfig)) {
        if(!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error() << std::endl;
        }
        return nullptr;
	}

    if(!reader.Warning().empty()) {
        std::cerr << "TinyObjReader: " << reader.Warning() << std::endl;
	}

	auto& attrib = reader.GetAttrib();
	auto& shapes = reader.GetShapes();
	auto& objMaterials = reader.GetMaterials();

    std::vector<cudaTextureObject_t> loadedTextures;
    std::vector<cudaArray_t> loadedTextureArrays;
    std::unordered_map<std::string, cudaTextureObject_t> textureCache;

    std::vector<Material> materials;
    std::vector<Geom> geometry;
    std::vector<Triangle> triangles;

    // Parse materials from obj
    for (const auto& objMaterial : objMaterials) {
        Material mat{};
        
        mat.albedo = glm::vec3(objMaterial.diffuse[0], objMaterial.diffuse[1], objMaterial.diffuse[2]);
		mat.blinnPhong.specularColor = glm::vec3(objMaterial.specular[0], objMaterial.specular[1], objMaterial.specular[2]);
		mat.blinnPhong.exponent = glm::clamp(objMaterial.shininess, 0.0f, 1000.0f);
        mat.ior = glm::max(1.0f, objMaterial.ior);

        mat.albedoTexture = 0;

        if (!objMaterial.diffuse_texname.empty()) {
            std::string fullTexPath = baseDir + objMaterial.diffuse_texname;

            auto iter = textureCache.find(fullTexPath);
            if (iter != textureCache.end()) {
                // Texture already loaded by another material, reuse the index
                mat.albedoTexture = iter->second;
            }
            else {
                // New unique texture, load via your helper class
                cudaTextureObject_t textureObject = 0;
                cudaArray_t textureArray = nullptr;

                if (Texture::LoadTexture(fullTexPath.c_str(), textureObject, textureArray)) {
                    loadedTextures.push_back(textureObject);
                    loadedTextureArrays.push_back(textureArray);
                    textureCache[fullTexPath] = textureObject;

                    mat.albedoTexture = textureObject;
                }
            }
        }

		float emissiveSum = objMaterial.emission[0] + objMaterial.emission[1] + objMaterial.emission[2];
        if(emissiveSum > 0.0f) {
            mat.type = MaterialType::OpaqueDiffuse;
            mat.albedo = glm::vec3(objMaterial.emission[0], objMaterial.emission[1], objMaterial.emission[2]);
            mat.emittance = 1.0f;
            materials.push_back(mat);
            continue;
        }

        // Check for perfect glass
        if(objMaterial.illum == 4 || objMaterial.illum == 9 || objMaterial.dissolve < 1.0f ||
            objMaterial.transmittance[0] > 0.0f || objMaterial.transmittance[1] > 0.0f || objMaterial.transmittance[2] > 0.0f) {
            mat.type = MaterialType::PerfectSpecular;
            mat.blinnPhong.bRefractive = true;
            if(mat.ior <= 1.0f) {
                mat.ior = 1.5f; // Default IOR for glass if not specified
			}
            materials.push_back(mat);
            continue;
		}

        // Check for perfect mirror
        if (objMaterial.illum == 3) {
            mat.type = MaterialType::PerfectSpecular;
            mat.blinnPhong.bRefractive = false;
            materials.push_back(mat);
			continue;
        }

        bool bSpecular = (mat.blinnPhong.specularColor.r > 0.0f ||
            mat.blinnPhong.specularColor.g > 0.0f ||
			mat.blinnPhong.specularColor.b > 0.0f);

        if(bSpecular && objMaterial.shininess > 0.0f) {
            mat.type = MaterialType::BlinnPhong;
            mat.blinnPhong.bRefractive = false;
            materials.push_back(mat);
            continue;
		}
        else {
			mat.type = MaterialType::OpaqueDiffuse;
        }

        materials.push_back(mat);
    }

    if (materials.empty()) {
		Material diffuse = Material{};
        Material glass = Material{};
		glass.type = MaterialType::PerfectSpecular;
		glass.albedo = glm::vec3(1.0f);
		glass.ior = 1.5f;
		glass.blinnPhong.bRefractive = true;
		glass.albedoTexture = -1;
        materials.push_back(diffuse);
    }

    // Parse meshes
    for (const auto& shape : shapes) {
        Geom geom{};
        geom.type = GeomType::MESH;

		geom.triangleOffset = triangles.size();
        size_t indexOffset = 0;

        // Iterate over polygons which should be triangles
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
			size_t fv = size_t(shape.mesh.num_face_vertices[f]);

			Triangle tri{};
			tinyobj::index_t idx0 = shape.mesh.indices[indexOffset + 0];
            tri.v0 = glm::vec3(
                attrib.vertices[3 * size_t(idx0.vertex_index) + 0],
                attrib.vertices[3 * size_t(idx0.vertex_index) + 1],
                attrib.vertices[3 * size_t(idx0.vertex_index) + 2]
			);
            if(idx0.texcoord_index >= 0) {
                tri.uv0 = glm::vec2(
                    attrib.texcoords[2 * size_t(idx0.texcoord_index) + 0],
                    attrib.texcoords[2 * size_t(idx0.texcoord_index) + 1]
                );
			}
            else {
				tri.uv0 = glm::vec2(0.0f, 0.0f);
            }

			tinyobj::index_t idx1 = shape.mesh.indices[indexOffset + 1];
            tri.v1 = glm::vec3(
                attrib.vertices[3 * size_t(idx1.vertex_index) + 0],
                attrib.vertices[3 * size_t(idx1.vertex_index) + 1],
                attrib.vertices[3 * size_t(idx1.vertex_index) + 2]
            );
            if(idx1.texcoord_index >= 0) {
                tri.uv1 = glm::vec2(
                    attrib.texcoords[2 * size_t(idx1.texcoord_index) + 0],
                    attrib.texcoords[2 * size_t(idx1.texcoord_index) + 1]
                );
            }
            else {
                tri.uv1 = glm::vec2(0.0f, 0.0f);
			}

			tinyobj::index_t idx2 = shape.mesh.indices[indexOffset + 2];
            tri.v2 = glm::vec3(
                attrib.vertices[3 * size_t(idx2.vertex_index) + 0],
                attrib.vertices[3 * size_t(idx2.vertex_index) + 1],
                attrib.vertices[3 * size_t(idx2.vertex_index) + 2]
            );
            if (idx2.texcoord_index >= 0) {
                tri.uv2 = glm::vec2(
                    attrib.texcoords[2 * size_t(idx2.texcoord_index) + 0],
                    attrib.texcoords[2 * size_t(idx2.texcoord_index) + 1]
                );
            }
            else {
                tri.uv2 = glm::vec2(0.0f, 0.0f);
			}

            triangles.push_back(tri);

			int objMaterialId = shape.mesh.material_ids[f];
			geom.materialid = (objMaterialId >= 0 && objMaterialId < materials.size()) ? objMaterialId : 0;

			indexOffset += fv;
        }

		geom.triangleCount = triangles.size() - geom.triangleOffset;

		geom.transform = glm::mat4(1.0f);
		geom.inverseTransform = glm::mat4(1.0f);
		geom.invTranspose = glm::mat4(1.0f);

        if (geom.triangleCount > 0) {
            geometry.push_back(geom);
        }
    }

    // Build BLAS
	std::vector<BLASNode> blasNodes;
	std::vector<Triangle> sortedTriangles;

    for (size_t g = 0; g < geometry.size(); g++) {
		Geom& geom = geometry[g];
		std::vector<TriangleBVHBuildData> triangleBuildData(geom.triangleCount);
        for (int t = 0; t < geom.triangleCount; t++) {
            const Triangle& tri = triangles[geom.triangleOffset + t];
            AABB bounds = AABB::fromTriangle(tri);
			triangleBuildData[t].bounds = bounds;
            triangleBuildData[t].centroid = (tri.v0 + tri.v1 + tri.v2) / 3.0f;
			triangleBuildData[t].triangleIndex = geom.triangleOffset + t;
        }

		geom.triangleOffset = sortedTriangles.size();

        std::vector<BLASNode> geomBlasNodes;
		BVHBuilder::buildBLAS(triangleBuildData, 0, triangleBuildData.size(), geomBlasNodes, triangles, sortedTriangles);
       
		geom.triangleCount = sortedTriangles.size() - geom.triangleOffset;

        int nodeOffsetStart = blasNodes.size();
        for (size_t n = 0; n < geomBlasNodes.size(); n++) {
            BLASNode& node = geomBlasNodes[n];
            if (node.triangleCount == 0) {
                node.leftChild += nodeOffsetStart;
            }
        }
        geom.blasNodeOffset = nodeOffsetStart;
		blasNodes.insert(blasNodes.end(), geomBlasNodes.begin(), geomBlasNodes.end());
    }

    // Sort geometry so lights are first BEFORE building TLAS
    auto it = std::partition(geometry.begin(), geometry.end(), [materials](const Geom& geom) {
        return geom.materialid < materials.size() && geom.materialid >= 0 && materials[geom.materialid].emittance > 0.0f;
        });
    int lightCount = std::distance(geometry.begin(), it);

    // Build TLAS
    std::vector<TLASNode> tlasNodes;

    std::vector<GeometryBVHBuildData> geometryBuildData(geometry.size());
    for (size_t g = 0; g < geometry.size(); g++) {
        Geom& geom = geometry[g];
		AABB bounds = AABB::fromGeometry(geom, blasNodes);
		geometryBuildData[g].bounds = bounds;
		geometryBuildData[g].centroid = (bounds.min + bounds.max) * 0.5f;
		geometryBuildData[g].geometryIndex = g;
    }

	BVHBuilder::buildTLAS(geometryBuildData, 0, geometryBuildData.size(), tlasNodes);

    return std::make_unique<Scene>(geometry, lightCount, sortedTriangles, blasNodes, tlasNodes, materials, loadedTextures, loadedTextureArrays);
}
