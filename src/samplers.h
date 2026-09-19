#pragma once

#include <cuda_runtime.h>

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

private:
	__host__ __device__ static glm::vec3 sampleLocalUniformHemisphere(const glm::vec2& random) {
		float z = random.x;
		float r = glm::sqrt(glm::max(0.0f, 1.0f - z * z));
		float phi = 2.0f * glm::pi<float>() * random.y;
		return glm::vec3(r * glm::cos(phi), r * glm::sin(phi), z);
	}
};