#include "material.h"

#include "samplers.h"

#include <limits>

__device__ glm::vec3 Material::pickOugoingDirection(
	const glm::vec3& normal, 
	const glm::vec3& wi, 
	const glm::vec2& random, 
	bool bInside,
	float& outPdf)
{
	if (isSpecular()) {
		return pickPerfectSpecularOutgoingDirection(normal, wi, random.x, bInside, outPdf);
	}
	else if (isGlossy()) {
		return pickGlossySpecularOutgoingDirection(normal, wi, random, outPdf);
	}
	else {
		return pickDiffuseOutgoingDirection(normal, random, outPdf);
	}
}

__device__ glm::vec3 Material::evaluateBrdf(const glm::vec3& normal, const glm::vec3& wi, const glm::vec3& wo, bool bDirectionGeneratedFromBrdf)
{
	if (isSpecular()) {
		return evaluatePerfectSpecularBrdf(bDirectionGeneratedFromBrdf);
	}
	else if (isGlossy()) {
		return evaluateGlossySpecularBrdf(normal, wi, wo);
	}
	else {
		return evaluateDiffuseBrdf(normal, wo);
	}
}

__device__ glm::vec3 Material::evaluateDiffuseBrdf(
	const glm::vec3& normal,
	const glm::vec3& outgoingDirection)
{
	glm::vec3 brdf = color / glm::pi<float>();
	float cosTheta = glm::max(0.0f, glm::dot(normal, outgoingDirection));

	return brdf * cosTheta;
}

__device__ glm::vec3 Material::pickDiffuseOutgoingDirection(const glm::vec3& normal, const glm::vec2& random, float& outPdf)
{
	return Samplers::sampleCosineWeightedHemisphere(normal, random, outPdf);
}

__device__ glm::vec3 Material::evaluateGlossySpecularBrdf(
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
	float D = ((specular.exponent + 2.0f) / (2.0f * glm::pi<float>())) * glm::pow(dotNH, specular.exponent);

	// Fresnel term
	glm::vec3 F = specular.color + (glm::vec3(1.0f) - specular.color) * glm::pow(1.0f - dotLH, 5.0f);

	// Geometry term
	float G = glm::min(1.0f, glm::min(
		(2.0f * dotNH * dotNV) / dotLH,
		(2.0f * dotNH * cosTheta) / dotLH
	));

	glm::vec3 brdfSpecular = (D * F * G) / (4.0f * dotNV * cosTheta);

	// Add energy conserved diffuse
	glm::vec3 brdfDiffuse = (glm::vec3(1.0f) - F) * (color / glm::pi<float>());

	return (brdfSpecular + brdfDiffuse) * cosTheta;
}

__device__ glm::vec3 Material::pickGlossySpecularOutgoingDirection(
	const glm::vec3& normal, 
	const glm::vec3& wi, 
	const glm::vec2& random, 
	float& outPdf)
{
	return Samplers::sampleWorldBlinnPhong(
		normal,
		wi,
		random,
		specular.exponent,
		outPdf
	);
}

__device__ glm::vec3 Material::evaluatePerfectSpecularBrdf(bool bDirectionGeneratedFromBrdf)
{
	if (bDirectionGeneratedFromBrdf) {
		return color;
	}
	else {
		return glm::vec3(0.0f);
	}
}

__device__ glm::vec3 Material::pickPerfectSpecularOutgoingDirection(const glm::vec3& normal, const glm::vec3& wi, float random, bool bInside, float& outPdf)
{
	// Flip direction for glm reflect + refract
	glm::vec3 incident = -wi;
	glm::vec3 hitNormal = normal;

	// Entering vs. exiting
	float iorIn = 1.0f;
	float iorOut = indexOfRefraction;

	if (bInside) {
		// Going from inside to out
		iorIn = indexOfRefraction;
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

	outPdf = std::numeric_limits<float>::infinity();

	if (hasRefractive > 0.0f && !bReflect) {
		return refracted;
	}
	else {
		return glm::reflect(incident, hitNormal);
	}
}
