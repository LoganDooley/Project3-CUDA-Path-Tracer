#include "shadingMaterial.h"

#include "samplers.h"

__device__ glm::vec3 ShadingMaterial::evaluateDiffuseMaterial(const Material& material, const glm::vec3& normal, const glm::vec2& random, glm::vec3& outOutgoingDirection)
{
	float pdf = 0.0f;
	outOutgoingDirection = Samplers::sampleCosineWeightedHemisphere(normal, random, pdf);

	glm::vec3 brdf = material.color / glm::pi<float>();
	float cosTheta = glm::max(0.0f, glm::dot(normal, outOutgoingDirection));
	
	// TODO: Simplify this to just material.color likely
	return (brdf * cosTheta) / pdf;
}

__device__ glm::vec3 ShadingMaterial::evaluateGlossySpecularMaterial(const Material& material, const glm::vec3& normal, const glm::vec3& wi, const glm::vec2& random, glm::vec3& outOutgoingDirection)
{
	float pdf = 0.0f;
	outOutgoingDirection = Samplers::sampleWorldBlinnPhong(
		normal,
		wi,
		random,
		material.specular.exponent,
		pdf
	);

	float cosTheta = glm::max(0.0f, glm::dot(normal, outOutgoingDirection));

	if (pdf <= 0.0f || cosTheta <= 0.0f) {
		return glm::vec3(0.0f);
	}

	// Evaluate blinn phong brdf
	glm::vec3 halfVector = glm::normalize(wi + outOutgoingDirection);
	float dotNH = glm::max(0.0f, glm::dot(normal, halfVector));
	float dotLH = glm::max(0.0f, glm::dot(outOutgoingDirection, halfVector));
	float dotNV = glm::max(0.0f, glm::dot(normal, wi));

	if (dotNV <= 0.0f) {
		return glm::vec3(0.0f);
	}

	// Blinn phong terms
	// Distribution term
	float D = ((material.specular.exponent + 2.0f) / (2.0f * glm::pi<float>())) * glm::pow(dotNH, material.specular.exponent);

	// Fresnel term
	glm::vec3 F = material.specular.color + (glm::vec3(1.0f) - material.specular.color) * glm::pow(1.0f - dotLH, 5.0f);

	// Geometry term
	float G = glm::min(1.0f, glm::min(
		(2.0f * dotNH * dotNV) / dotLH,
		(2.0f * dotNH * cosTheta) / dotLH
	));

	glm::vec3 brdfSpecular = (D * F * G) / (4.0f * dotNV * cosTheta);

	// Add energy conserved diffuse
	glm::vec3 brdfDiffuse = (glm::vec3(1.0f) - F) * (material.color / glm::pi<float>());

	return ((brdfSpecular + brdfDiffuse) * cosTheta) / pdf;
}
