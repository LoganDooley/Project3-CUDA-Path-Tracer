#pragma once

#include "pathTraceCommon.h"

struct DevScene {
	__device__ IntersectionData intersect(const Ray& ray);

	Geom* dev_geometry = nullptr;
	size_t m_geometryCount = 0;
	Material* dev_materials = nullptr;
	size_t m_materialCount = 0;
};