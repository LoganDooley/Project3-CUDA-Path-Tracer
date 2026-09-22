#pragma once

#include <cuda_runtime.h>

#include <glm/gtc/constants.hpp>

#include "mathHelpers.h"

class Samplers {
public:
	__host__ __device__ static glm::vec3 sampleWorldUniformHemisphere(const glm::vec3& normal, const glm::vec2& random, float& outPdf){
		glm::vec3 localSample = sampleLocalUniformHemisphere(random);

		glm::vec3 tangent;
		glm::vec3 bitangent;
		MathHelpers::createCoordinateSystem(normal, tangent, bitangent);

		outPdf = 1.0f / (2.0f * glm::pi<float>());

		return localSample.x * tangent + localSample.y * bitangent + localSample.z * normal;
	}

	__host__ __device__ static glm::vec3 sampleCosineWeightedHemisphere(const glm::vec3& normal, const glm::vec2& random, float& outPdf) {
		glm::vec3 localSample = sampleLocalCosineWeightedHemisphere(random);

		glm::vec3 tangent;
		glm::vec3 bitangent;
		MathHelpers::createCoordinateSystem(normal, tangent, bitangent);

		glm::vec3 outgoingDirection = localSample.x * tangent + localSample.y * bitangent + localSample.z * normal;

		float cosTheta = glm::max(0.0f, glm::dot(normal, outgoingDirection));
		outPdf = cosTheta / glm::pi<float>();

		return outgoingDirection;
	}

	__host__ __device__ static glm::vec3 sampleWorldBlinnPhong(const glm::vec3& normal,
		const glm::vec3& incoming,
		const glm::vec2& random,
		float specularExponent,
		float& outPdf) {
		glm::vec3 localHalfVector = sampleLocalBlinnPhong(random, specularExponent);

		glm::vec3 tangent;
		glm::vec3 bitangent;
		MathHelpers::createCoordinateSystem(normal, tangent, bitangent);
		glm::vec3 halfVector = localHalfVector.x * tangent + localHalfVector.y * bitangent + localHalfVector.z * normal;

		// Make sure h is in the same hemisphere as the normal
		if (glm::dot(halfVector, normal) < 0.0f) {
			halfVector = -halfVector;
		}

		glm::vec3 outgoing = 2.0f * glm::dot(incoming, halfVector) * halfVector - incoming;

		// Calculate the pdf
		float dotNH = glm::max(0.0f, glm::dot(normal, halfVector));
		float dotIH = glm::max(0.0f, glm::dot(incoming, halfVector));

		if (dotIH <= 0.0f) {
			// This should flag to termiante the path, invalid
			outPdf = 0.0f;
			return glm::vec3(0.0f);
		}

		float dH = ((specularExponent + 2.0f) / (2.0f * glm::pi<float>())) * glm::pow(dotNH, specularExponent);
		outPdf = dH / (4.0f * dotIH);
		return outgoing;
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

	__host__ __device__ static glm::vec3 sampleLocalBlinnPhong(const glm::vec2& random, float specularExponent) {
		float phi = 2.0f * glm::pi<float>() * random.y;
		float cosTheta = glm::pow(random.x, 1.0f / (specularExponent + 1.0f));
		float sinTheta = glm::sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));

		return glm::vec3(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi), cosTheta);
	}
};