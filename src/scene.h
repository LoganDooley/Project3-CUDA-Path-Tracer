#pragma once

#include "pathTraceCommon.h"

class Scene {
public:
	Scene();
	~Scene();

	const Scene& operator=(const Scene&) = delete;
	Scene(const Scene&) = delete;

	const Scene& operator=(Scene&&) noexcept;
	Scene(Scene&&) noexcept;

private:
	Geom* dev_geometry;
	Material* dev_materials;
};