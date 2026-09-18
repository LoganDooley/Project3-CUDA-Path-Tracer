#pragma once

#include <glm/glm.hpp>

#include "mathHelpers.h"

struct Ray {
public:
	__host__ __device__ glm::vec3 getPositionAtTime(const float& t) const {
		return origin + t * direction;
	}

	glm::vec3 origin = glm::vec3(0.0f);
	glm::vec3 direction = glm::vec3(1.0f, 0.0f, 0.0f);
};

struct PathState {
	Ray ray = Ray{};
	glm::vec3 throughput = glm::vec3(1.0f);
	glm::vec3 accumulatedColor = glm::vec3(0.0f);
	int bounceCount = 0;
	bool active = true;
	int pixelIndex = -1;
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

		float t0;
		float t1;

		glm::vec3 toRay = ray.origin - position;
		float a = glm::dot(ray.direction, ray.direction);
		float b = 2 * glm::dot(ray.direction, toRay);
		float c = glm::dot(toRay, toRay) - radius * radius;
		if (!MathHelpers::solveQuadratic(a, b, c, t0, t1)) {
			return result;
		}
		if (t0 > t1) {
			float temp = t1;
			t1 = t0;
			t0 = temp;
		}

		if (t0 < 0.0f) {
			t0 = t1;
			if (t0 < 0.0f) {
				return result;
			}
		}

		result.bHit = true;
		result.position = ray.getPositionAtTime(t0);
		result.normal = MathHelpers::safeNormalize(result.position - position);

		return result;
	}

	glm::vec3 position = glm::vec3(0.0f);
	float radius = 1.f;
};