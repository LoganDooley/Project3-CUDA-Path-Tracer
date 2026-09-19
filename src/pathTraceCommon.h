#pragma once

#include <glm/glm.hpp>

#include "mathHelpers.h"

struct Ray {
public:
	__host__ __device__ glm::vec3 getPositionAtTime(const float& t) const {
		return origin + t * direction;
	}

	__host__ __device__ Ray transform(const glm::mat4& transform) const {
		Ray transformed;
		transformed.origin = glm::vec3(transform * glm::vec4(origin, 1.0f));
		transformed.direction = glm::vec3(transform * glm::vec4(direction, 0.0f));
		return transformed;
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

enum GeomType
{
	SPHERE,
	CUBE
};

struct Geom
{
	enum GeomType type = GeomType::SPHERE;
	int materialid = 0;

	// TODO: Remove vec3s from device since these should be stored in the mat4s and are redundant
	glm::vec3 translation = glm::vec3(0.0f);
	glm::vec3 rotation = glm::vec3(0.0f);
	glm::vec3 scale = glm::vec3(1.0f);

	glm::mat4 transform = glm::mat4(1.0f);
	glm::mat4 inverseTransform = glm::mat4(1.0f);
	glm::mat4 invTranspose = glm::mat4(1.0f);
};

struct Material
{
	glm::vec3 color = glm::vec3(1.0, 0.0, 0.203);
	struct
	{
		float exponent = 1.f;
		glm::vec3 color = glm::vec3(0.0f);
	} specular;
	float hasReflective = -1.f;
	float hasRefractive = -1.f;
	float indexOfRefraction = 1.f;
	float emittance = 0.f;
};

struct IntersectionData {
	glm::vec3 normal = glm::vec3(1, 0, 0);
	float t = -1;
	int materialIndex = 0;
};

class IntersectionStatics {
public:
	__device__ static IntersectionData intersectGeometry(const Ray& ray, const Geom& geometry) {
		// Convert world space -> object spce
		Ray objectSpaceRay = ray.transform(geometry.inverseTransform);
		
		// Run intersection
		IntersectionData result = intersectSphere(objectSpaceRay);
		
		if (result.t <= 0.0f) {
			return result;
		}

		result.materialIndex = geometry.materialid;

		// Convert normal to world space
		glm::vec4 worldNormal = geometry.invTranspose * glm::vec4(result.normal, 0.0f);
		result.normal = MathHelpers::safeNormalize(glm::vec3(worldNormal));

		// Convert t to world space
		glm::vec3 worldPosition = glm::vec3(geometry.transform * glm::vec4(objectSpaceRay.getPositionAtTime(result.t), 1.0f));
		result.t = glm::length(worldPosition - ray.origin);

		return result;
	}

private:
	__device__ static IntersectionData intersectSphere(const Ray& ray) {
		IntersectionData result = IntersectionData{};

		float t0;
		float t1;

		glm::vec3 toRay = ray.origin;
		float a = glm::dot(ray.direction, ray.direction);
		float b = 2 * glm::dot(ray.direction, toRay);
		float c = glm::dot(toRay, toRay) - 0.5f * 0.5f;
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

		result.normal = MathHelpers::safeNormalize(ray.getPositionAtTime(t0));
		result.t = t0;

		return result;
	}
};