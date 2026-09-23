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
	float previousPdf = 0.0f;
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
	float t = -1.0f;
	int materialIndex = 0;
	bool bInside = false;
};

class IntersectionStatics {
public:
	__device__ static IntersectionData intersectGeometry(const Ray& ray, const Geom& geometry) {
		// Convert world space -> object spce
		Ray objectSpaceRay = ray.transform(geometry.inverseTransform);
		
		// Run intersection
		IntersectionData result;
		if (geometry.type == GeomType::SPHERE) {
			result = intersectSphere(objectSpaceRay);
		}
		else {
			result = intersectBox(objectSpaceRay);
		}
		
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
			result.bInside = true;
		}
		else {
			result.bInside = false;
		}

		result.normal = MathHelpers::safeNormalize(ray.getPositionAtTime(t0));
		result.t = t0;

		// Flip the normal if we were inside
		float normalSign = result.bInside ? -1.0f : 1.0f;
		result.normal *= normalSign;

		return result;
	}

	__device__ static IntersectionData intersectBox(const Ray& ray) {
		IntersectionData result = IntersectionData{};

		glm::vec3 boxMin(-0.5f);
		glm::vec3 boxMax(0.5f);

		// Using slab method for fast aabb
		glm::vec3 invDirection = 1.0f / ray.direction;

		glm::vec3 tMin = (boxMin - ray.origin) * invDirection;
		glm::vec3 tMax = (boxMax - ray.origin) * invDirection;

		glm::vec3 tNear = (glm::min)(tMin, tMax);
		glm::vec3 tFar = (glm::max)(tMin, tMax);

		float t0 = (glm::max)(tNear.x, (glm::max)(tNear.y, tNear.z));
		float t1 = (glm::min)(tFar.x, (glm::min)(tFar.y, tFar.z));

		// Check for miss
		if (t0 > t1 || t1 < 0.0f) {
			return result;
		}

		if (t0 < 0.0f) {
			result.t = t1;
			result.bInside = true;
		}
		else {
			result.t = t0;
			result.bInside = false;
		}

		if (result.t < 0.0f) {
			return result;
		}

		result.normal = glm::vec3(0.0f);
		if (result.t == tNear.x) {
			result.normal.x = (ray.direction.x > 0.0f) ? -1.0f : 1.0f;
		}
		else if (result.t == tNear.y) {
			result.normal.y = (ray.direction.y > 0.0f) ? -1.0f : 1.0f;
		}
		else {
			result.normal.z = (ray.direction.z > 0.0f) ? -1.0f : 1.0f;
		}

		// Flip the normal if we were inside
		float normalSign = result.bInside ? -1.0f : 1.0f;
		result.normal *= normalSign;

		return result;
	}
};