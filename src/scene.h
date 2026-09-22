#pragma once

#include "pathTraceCommon.h"

#include "devScene.h"

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
			dev_materials,
			m_materialCount
		};
	}

	Geom* dev_geometry = nullptr;
	size_t m_geometryCount = 0;

	Material* dev_materials = nullptr;
	size_t m_materialCount = 0;
};

class SceneLoader {
public:
	static std::unique_ptr<Scene> loadFromFile(const std::string& filepath);
};