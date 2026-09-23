#pragma once

#include "pathTraceCommon.h"

struct DevScene {
	__device__ IntersectionData intersect(const Ray& ray);

	__device__ bool isVisible(const Ray& ray, float tMax);

	__device__ glm::vec3 nextEventEsimation(const glm::vec4& random, const Ray& incomingRay, const IntersectionData& intersectionData, float& outPdf);

	Geom* dev_geometry = nullptr;
	size_t m_geometryCount = 0;
	Material* dev_materials = nullptr;
	size_t m_materialCount = 0;
	size_t m_lightCount = 0;
};