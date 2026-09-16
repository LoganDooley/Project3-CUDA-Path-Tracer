#pragma once

#include <glm/glm.hpp>

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
	int pixelIndex;
};

struct PathState {
	glm::vec3 throughput = glm::vec3(1.0f);
	glm::vec3 accumulatedColor = glm::vec3(0.0f);
	int bounceCount = 0;
	bool active = true;
};

struct IntersectionData {
	glm::vec3 position = glm::vec3(0);
	int materialIndex = 0;
	glm::vec3 normal = glm::vec3(1, 0, 0);
	bool bHit = false;
};

struct Primitive {
	__device__ virtual IntersectionData intersect(const Ray& ray) 
	{
		return IntersectionData{};
	}
};

struct Sphere : public Primitive {
	__device__ IntersectionData intersect(const Ray& ray) override
	{
		IntersectionData result = IntersectionData{};
		result.bHit = true;
		return result;
	}

	glm::vec3 position;
	float radius;
};