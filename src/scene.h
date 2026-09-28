#pragma once

#include "pathTraceCommon.h"

#include "devScene.h"
#include "bvh.h"

#include <string>
#include <memory>

class Scene {
public:
	Scene();
	~Scene();

	Scene& operator=(const Scene&) = delete;
	Scene(const Scene&) = delete;

	Scene& operator=(Scene&&) noexcept;
	Scene(Scene&&) noexcept;

	DevScene getDevScene() {
		return DevScene{
			dev_geometry,
			m_geometryCount,
			dev_triangles,
			m_triangleCount,
			dev_blasNodes,
			m_blasNodeCount,
			dev_materials,
			m_materialCount,
			m_lightCount
		};
	}

	Geom* dev_geometry = nullptr;
	size_t m_geometryCount = 0;

	Triangle* dev_triangles = nullptr;
	size_t m_triangleCount = 0;

	BLASNode* dev_blasNodes = nullptr;
	size_t m_blasNodeCount = 0;

	Material* dev_materials = nullptr;
	size_t m_materialCount = 0;

	// First m_lightCount pieces of geometry are lights
	size_t m_lightCount = 0;
};

class SceneLoader {
public:
	static std::unique_ptr<Scene> loadFromFile(const std::string& filepath);

private:
	static std::unique_ptr<Scene> loadFromJson(const std::string& filepath);
	static std::unique_ptr<Scene> loadFromGltf(const std::string& filepath);
	static std::unique_ptr<Scene> loadFromObj(const std::string& filepath);
};