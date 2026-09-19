#pragma once

#include "pathTraceCommon.h"

#include <string>
#include <memory>

class Scene {
public:
	Scene();
	~Scene();

	const Scene& operator=(const Scene&) = delete;
	Scene(const Scene&) = delete;

	const Scene& operator=(Scene&&) noexcept;
	Scene(Scene&&) noexcept;

	Geom* dev_geometry = nullptr;
	size_t m_geometryCount = 0;

	Material* dev_materials = nullptr;
	size_t m_materialCount = 0;
};

class SceneLoader {
public:
	static std::unique_ptr<Scene> loadFromFile(const std::string& filepath);
};