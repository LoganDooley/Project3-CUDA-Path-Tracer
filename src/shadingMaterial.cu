#include "shadingMaterial.h"

#include "samplers.h"

__device__ glm::vec3 ShadingMaterial::evaluateDiffuseMaterial(const Material& material,
	const glm::vec3& normal,
	const glm::vec3& outgoingDirection)
{
	glm::vec3 brdf = material.color / glm::pi<float>();
	float cosTheta = glm::max(0.0f, glm::dot(normal, outgoingDirection));

	return brdf * cosTheta;
}

__device__ glm::vec3 ShadingMaterial::pickDiffuseOutgoingDirection(const glm::vec3& normal, const glm::vec2& random, float& outPdf)
{
	return Samplers::sampleCosineWeightedHemisphere(normal, random, outPdf);
}

__device__ glm::vec3 ShadingMaterial::evaluateGlossySpecularMaterial(const Material& material,
	const glm::vec3& normal,
	const glm::vec3& wi,
	const glm::vec3& outgoingDirection)
{
	float cosTheta = glm::max(0.0f, glm::dot(normal, outgoingDirection));

	if (cosTheta <= 0.0f) {
		return glm::vec3(0.0f);
	}

	// Evaluate blinn phong brdf
	glm::vec3 halfVector = glm::normalize(wi + outgoingDirection);
	float dotNH = glm::max(0.0f, glm::dot(normal, halfVector));
	float dotLH = glm::max(0.0f, glm::dot(outgoingDirection, halfVector));
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

	return (brdfSpecular + brdfDiffuse) * cosTheta;
}

__device__ glm::vec3 ShadingMaterial::pickGlossySpecularOutgoingDirection(const Material& material, const glm::vec3& normal, const glm::vec3& wi, const glm::vec2& random, float& outPdf)
{
	return Samplers::sampleWorldBlinnPhong(
		normal,
		wi,
		random,
		material.specular.exponent,
		outPdf
	);

}

__device__ glm::vec3 ShadingMaterial::evaluatePerfectSpecularMaterial(const Material& material, const glm::vec3& normal, const glm::vec3& wi, float random, bool bInside, glm::vec3& outOutgoingDirection, float& outEpsilonSign)
{
	// Flip direction for glm reflect + refract
	glm::vec3 incident = -wi;
	glm::vec3 hitNormal = normal;

	// Entering vs. exiting
	float iorIn = 1.0f;
	float iorOut = material.indexOfRefraction;

	if (bInside) {
		// Going from inside to out
		iorIn = material.indexOfRefraction;
		iorOut = 1.0f;
	}

	float iorRatio = iorIn / iorOut;
	float dotNV = glm::max(0.0f, glm::dot(hitNormal, wi));

	float r0 = (iorIn - iorOut) / (iorIn + iorOut);
	r0 = r0 * r0;
	float fresnel = r0 + (1.0f - r0) * glm::pow(1.0f - dotNV, 5.0f);

	// Refract
	glm::vec3 refracted = glm::refract(incident, hitNormal, iorRatio);

	bool bTotalInternalReflection = (glm::dot(refracted, refracted) <= 0.0f);
	bool bReflect = (random < fresnel) || bTotalInternalReflection;

	if (material.hasRefractive > 0.0f && !bReflect) {
		outOutgoingDirection = refracted;

		outEpsilonSign = -1.0f;
	}
	else {
		outOutgoingDirection = glm::reflect(incident, hitNormal);

		outEpsilonSign = 1.0f;
	}

	return material.color;
}
