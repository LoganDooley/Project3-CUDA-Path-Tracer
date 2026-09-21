#pragma once

#include <cuda_runtime.h>

#include <glm/gtc/constants.hpp>

#include "mathHelpers.h"

class Samplers {
public:
	__host__ __device__ static glm::vec3 sampleWorldUniformHemisphere(const glm::vec3& normal, const glm::vec2& random) {
		glm::vec3 localSample = sampleLocalUniformHemisphere(random);

		glm::vec3 tangent;
		glm::vec3 bitangent;
		MathHelpers::createCoordinateSystem(normal, tangent, bitangent);

		return localSample.x * tangent + localSample.y * bitangent + localSample.z * normal;
	}

	__host__ __device__ static glm::vec3 sampleCosineWeightedHemisphere(const glm::vec3& normal, const glm::vec2& random) {
		glm::vec3 localSample = sampleLocalCosineWeightedHemisphere(random);

		glm::vec3 tangent;
		glm::vec3 bitangent;
		MathHelpers::createCoordinateSystem(normal, tangent, bitangent);

		return localSample.x * tangent + localSample.y * bitangent + localSample.z * normal;
	}

private:
	__host__ __device__ static glm::vec3 sampleLocalUniformHemisphere(const glm::vec2& random) {
		float z = random.x;
		float r = glm::sqrt(glm::max(0.0f, 1.0f - z * z));
		float phi = 2.0f * glm::pi<float>() * random.y;
		return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), z);
	}

	__host__ __device__ static glm::vec2 sampleUniformDiskConcentric(const glm::vec2& random) {
		glm::vec2 uOffset = 2.0f * random - glm::vec2(1.f);
		if (uOffset.x == 0.0f && uOffset.y == 0.0f) {
			return glm::vec2(0.0f);
		}

		float theta;
		float r;
		if (glm::abs(uOffset.x) > glm::abs(uOffset.y)) {
			r = uOffset.x;
			theta = (glm::pi<float>() / 4.0f) * (uOffset.y / uOffset.x);
		}
		else {
			r = uOffset.y;
			theta = (glm::pi<float>() / 2.0f) - (glm::pi<float>() / 4.0f) * (uOffset.x / uOffset.y);
		}
		return r * glm::vec2(glm::cos(theta), glm::sin(theta));
	}

	__host__ __device__ static glm::vec3 sampleLocalCosineWeightedHemisphere(const glm::vec2& random) {
		glm::vec2 d = sampleUniformDiskConcentric(random);
		float z = glm::sqrt(1 - d.x * d.x - d.y * d.y);
		return glm::vec3(d.x, d.y, z);
	}
};