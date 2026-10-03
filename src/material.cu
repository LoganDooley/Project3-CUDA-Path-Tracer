#include "material.h"

#include "samplers.h"
#include "devScene.h"
#include "intersection.h"

#include <limits>

__device__ void Material::initializeFromIntersection(const IntersectionData& intersectionData)
{
	float u = intersectionData.uv.x;
	float v = intersectionData.uv.y;

	v = 1.0f - v;
	if(albedoTexture != 0) {
		float4 texColor = tex2D<float4>(albedoTexture, u, v);

		// Linearize albedo color
		albedo.x = powf(texColor.x, 2.2f);
		albedo.y = powf(texColor.y, 2.2f);
		albedo.z = powf(texColor.z, 2.2f);
	}

	if(type == MaterialType::PbrMetallicRoughness && pbr.metallicRoughnessTexture != 0) {
		float4 texColor = tex2D<float4>(pbr.metallicRoughnessTexture, u, v);

		pbr.metallic *= texColor.y;
		pbr.roughness *= texColor.z;
	}
}

__device__ glm::vec3 Material::evaluate(const glm::vec3& n, const glm::vec3& wi, const glm::vec3& wo, bool bInside, bool bDirectionGeneratedFromBrdf)
{
	if (isDelta()) {
		return glm::vec3(0.0f);
	}

	float cosThetaI = glm::dot(n, wi);
	float cosThetaO = glm::dot(n, wo);
	if (cosThetaI <= 0.0f) {
		return glm::vec3(0.0f);
	}

	bool bIsReflection = cosThetaO > 0.0f;
	float absCosThetaO = glm::abs(cosThetaO);

	if (type == MaterialType::OpaqueDiffuse) {
		if(!bIsReflection) {
			return glm::vec3(0.0f);
		}
		return albedo / glm::pi<float>();
	}

	MicrofacetScattering scattering = getSpecularScattering();
	glm::vec3 specularComponent = scattering.evaluate(n, wi, wo, bInside);

	if (type == MaterialType::PbrMetallicRoughness) {
		if (bIsReflection) {
			glm::vec3 H = glm::normalize(wi + wo);
			float dotIH = glm::max(0.0f, glm::dot(wi, H));
			float etaI = bInside ? ior : 1.0f;
			float etaT = bInside ? 1.0f : ior;
			float r0 = (etaI - etaT) / (etaI + etaT);
			r0 = r0 * r0;
			float F = r0 + (1.0f - r0) * glm::pow(1.0f - dotIH, 5.0f);

			glm::vec3 diffuseComponent = (1.0f - F) * (albedo / glm::pi<float>()) * (1.0f - pbr.metallic) * (1.0f - pbr.transmission);
			return diffuseComponent + specularComponent;
		}
		else {
			return specularComponent * pbr.transmission;
		}
	}
	else {
		if(!bIsReflection) {
			return glm::vec3(0.0f);
		}

		glm::vec3 diffuseComponent = (albedo / glm::pi<float>());
      	return diffuseComponent + specularComponent;
	}
}

__device__ float Material::pdf(const glm::vec3& n, const glm::vec3& wi, const glm::vec3& wo, bool bInside, bool bDirectionGeneratedFromBrdf)
{
	if(isDelta()) {
		return 0.0f;
	}

	float cosThetaI = glm::dot(n, wi);
	float cosThetaO = glm::dot(n, wo);
	if(cosThetaI <= 0.0f) {
		return 0.0f;
	}

	bool bIsReflection = cosThetaO > 0.0f;
	float absCosThetaO = glm::abs(cosThetaO);

	if (type == MaterialType::OpaqueDiffuse) {
		if(!bIsReflection) {
			return 0.0f;
		}
		return absCosThetaO / glm::pi<float>();
	}

	MicrofacetScattering scattering = getSpecularScattering();
	return scattering.pdf(n, wi, wo, bInside);
}

__device__ void Material::sample(
	const glm::vec3& n,
	const glm::vec3& wi,
	const glm::vec3& random,
	bool bInside,
	glm::vec3& wo,
	glm::vec3& outThroughput,
	float& outPdf,
	bool& bIsTransmission)
{
	float cosThetaI = glm::dot(n, wi);
	if (cosThetaI <= 0.0f) {
		outPdf = 0.0f;
		outThroughput = glm::vec3(0.0f);
		return;
	}

	if (isDelta()) {
		if (type == MaterialType::PerfectSpecular && !blinnPhong.bRefractive) {
			wo = glm::reflect(-wi, n);
			outPdf = 1.0f;
			outThroughput = albedo;
			bIsTransmission = false;
			return;
		}

		// Glass
		float etaI = bInside ? ior : 1.0f;
		float etaT = bInside ? 1.0f : ior;
		float r0 = (etaI - etaT) / (etaI + etaT);
		r0 = r0 * r0;
		float F = r0 + (1.0f - r0) * glm::pow(1.0f - cosThetaI, 5.0f);

		float eta = etaI / etaT;
		glm::vec3 refracted = glm::refract(-wi, n, eta);
		bool bTIR = (glm::dot(refracted, refracted) <= 0.0f);

		if(random.z < F || bTIR) {
			wo = glm::reflect(-wi, n);
			outPdf = bTIR ? 1.0f : F;

			if(type == MaterialType::PerfectSpecular) {
				outThroughput = albedo;
			}
			else {
				outThroughput = glm::mix(glm::vec3(F), albedo, pbr.metallic);
			}

			bIsTransmission = false;
		}
		else {
			wo = glm::normalize(refracted);
			outPdf = 1.0f - F;

			if(type == MaterialType::PerfectSpecular) {
				outThroughput = albedo;
			}
			else {
				outThroughput = albedo * pbr.transmission;
			}

			bIsTransmission = true;
		}

		outPdf = 1.0f;
		return;
	}

	if (type == MaterialType::OpaqueDiffuse) {
		bIsTransmission = false;

		wo = Samplers::sampleCosineWeightedHemisphere(n, glm::vec2(random), outPdf);
		outThroughput = albedo;
		return;
	}

	MicrofacetScattering scattering = getSpecularScattering();
	scattering.sample(n, wi, random, bInside, wo, outThroughput, outPdf, bIsTransmission);

	if (type == MaterialType::PbrMetallicRoughness && !bIsTransmission) {
		glm::vec3 evaluatedTotal = evaluate(n, wi, wo, bInside, true);
		outThroughput = (evaluatedTotal * glm::dot(n, wo)) / outPdf;
	}
}

__device__ glm::vec3 Material::sampleBrdf(
	const glm::vec3& normal, 
	const glm::vec3& wi, 
	const glm::vec2& random, 
	bool bInside,
	float& outPdf)
{
	if (type == MaterialType::PerfectSpecular) {
		return samplePerfectSpecularBrdf(normal, wi, random.x, bInside, outPdf);
	}
	else if (type == MaterialType::BlinnPhong) {
		return sampleGlossySpecularBrdf(normal, wi, random, outPdf);
	}
	else {
		return sampleDiffuseBrdf(normal, random, outPdf);
	}
}

__device__ float Material::getBrdfPdf(const glm::vec3& normal, const glm::vec3& wi, const glm::vec3& wo, bool bDirectionGeneratedFromBrdf)
{
	if(type == MaterialType::PerfectSpecular) {
		return bDirectionGeneratedFromBrdf ? 1.0f : 0.0f;
	}
	else if (type == MaterialType::BlinnPhong) {
		return Samplers::getBlinnPhongPdf(normal, wi, wo, blinnPhong.exponent);
	}
	else {
		return Samplers::getCosineWeightedHemispherePdf(normal, wo);
	}
}

__device__ glm::vec3 Material::evaluateBrdf(const glm::vec3& normal, const glm::vec3& wi, const glm::vec3& wo, bool bDirectionGeneratedFromBrdf)
{
	if (type == MaterialType::PerfectSpecular) {
		return evaluatePerfectSpecularBrdf(bDirectionGeneratedFromBrdf);
	}
	else if (type == MaterialType::BlinnPhong) {
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
	glm::vec3 brdf = albedo / glm::pi<float>();
	float cosTheta = glm::max(0.0f, glm::dot(normal, outgoingDirection));

	return brdf * cosTheta;
}

__device__ glm::vec3 Material::sampleDiffuseBrdf(const glm::vec3& normal, const glm::vec2& random, float& outPdf)
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
	float D = ((blinnPhong.exponent + 2.0f) / (2.0f * glm::pi<float>())) * glm::pow(dotNH, blinnPhong.exponent);

	// Fresnel term
	glm::vec3 F = blinnPhong.specularColor + (glm::vec3(1.0f) - blinnPhong.specularColor) * glm::pow(1.0f - dotLH, 5.0f);

	// Geometry term
	float G = glm::min(1.0f, glm::min(
		(2.0f * dotNH * dotNV) / dotLH,
		(2.0f * dotNH * cosTheta) / dotLH
	));

	glm::vec3 brdfSpecular = (D * F * G) / (4.0f * dotNV * cosTheta);

	// Add energy conserved diffuse
	glm::vec3 brdfDiffuse = (glm::vec3(1.0f) - F) * (albedo / glm::pi<float>());

	return (brdfSpecular + brdfDiffuse) * cosTheta;
}

__device__ glm::vec3 Material::sampleGlossySpecularBrdf(
	const glm::vec3& normal, 
	const glm::vec3& wi, 
	const glm::vec2& random, 
	float& outPdf)
{
	return Samplers::sampleWorldBlinnPhong(
		normal,
		wi,
		random,
		blinnPhong.exponent,
		outPdf
	);
}

__device__ glm::vec3 Material::evaluatePerfectSpecularBrdf(bool bDirectionGeneratedFromBrdf)
{
	if (bDirectionGeneratedFromBrdf) {
		return albedo;
	}
	else {
		return glm::vec3(0.0f);
	}
}

__device__ glm::vec3 Material::samplePerfectSpecularBrdf(const glm::vec3& normal, const glm::vec3& wi, float random, bool bInside, float& outPdf)
{
	// Flip direction for glm reflect + refract
	glm::vec3 incident = -wi;
	glm::vec3 hitNormal = normal;

	// Entering vs. exiting
	float iorIn = 1.0f;
	float iorOut = ior;

	if (bInside) {
		// Going from inside to out
		iorIn = ior;
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

	if (blinnPhong.bRefractive && !bReflect) {
		return refracted;
	}
	else {
		return glm::reflect(incident, hitNormal);
	}
}

__device__ glm::vec3 Material::evaluatePbrGGXBrdf(const glm::vec3& normal, const glm::vec3& wi, const glm::vec3& wo, bool bInside)
{
	float cosThetaO = glm::dot(normal, wo);
	float cosThetaI = glm::dot(normal, wi);

	// First determine if the light is reflected or transmitted
	bool bIsReflection = (cosThetaO * cosThetaI) > 0.0f;

	cosThetaO = glm::abs(cosThetaO);
	cosThetaI = glm::abs(cosThetaI);

	if(cosThetaO <= 0.0f || cosThetaI <= 0.0f) {
		return glm::vec3(0.0f);
	}

	float alpha = pbr.roughness * pbr.roughness;
	float alpha2 = alpha * alpha;

	if (bIsReflection) {
		// Evaluate BRDF
		glm::vec3 H = glm::normalize(wi + wo);
		float dotNH = glm::max(0.0f, glm::dot(normal, H));
		float dotHO = glm::max(0.0f, glm::dot(H, wo));

		float denomD = dotNH * dotNH * (alpha2 - 1.0f) + 1.0f;
		float D = alpha2 / (glm::pi<float>() * denomD * denomD);

		float etaI = bInside ? ior : 1.0f;
		float etaT = bInside ? 1.0f : ior;
		float r0 = (etaI - etaT) / (etaI + etaT);
		r0 = r0 * r0;

		glm::vec3 F0 = glm::mix(glm::vec3(r0), albedo, pbr.metallic);
		glm::vec3 F = F0 + (glm::vec3(1.0f) - F0) * glm::pow(1.0f - dotHO, 5.0f);

		auto SmithG1 = [](float cosTheta, float alpha) {
			float cosTheta2 = cosTheta * cosTheta;
			float tanTheta2 = (1.0f - cosTheta2) / cosTheta2;
			return 2.0f / (1.0f + glm::sqrt(1.0f + alpha * alpha * tanTheta2));
		};
		float G = SmithG1(cosThetaI, alpha) * SmithG1(cosThetaO, alpha);

		glm::vec3 specular = (D * F * G) / (4.0f * cosThetaI * cosThetaO);

		glm::vec3 diffuse = (glm::vec3(1.0f) - F) * (albedo / glm::pi<float>()) * (1.0f - pbr.metallic) * (1.0f - pbr.transmission);
	
		return (diffuse + specular) * cosThetaO;
	}
	else {
		// Evaluate BTDF
		if(pbr.transmission <= 0.0f) {
			return glm::vec3(0.0f);
		}

		float etaI = bInside ? ior : 1.0f;
		float etaT = bInside ? 1.0f : ior;

		glm::vec3 H = glm::normalize(wi + wo * (etaT / etaI));
		if(glm::dot(normal, H) < 0.0f) {
			H = -H;
		}

		float dotNH = glm::max(0.0f, glm::dot(normal, H));
		float dotIH = glm::max(0.0f, glm::dot(wi, H));
		float dotOH = glm::max(0.0f, glm::dot(wo, H));

		// D term
		float denomD = dotNH * dotNH * (alpha2 - 1.0f) + 1.0f;
		float D = alpha2 / (glm::pi<float>() * denomD * denomD);

		// G term
		auto SmithG1 = [](float cosTheta, float alpha) {
			float cosTheta2 = cosTheta * cosTheta;
			float tanTheta2 = (1.0f - cosTheta2) / cosTheta2;
			return 2.0f / (1.0f + glm::sqrt(1.0f + alpha * alpha * tanTheta2));
		};
		float G = SmithG1(cosThetaI, alpha) * SmithG1(cosThetaO, alpha);

		// F term
		float r0 = (etaI - etaT) / (etaI + etaT);
		r0 = r0 * r0;
		float F = r0 + (1.0f - r0) * glm::pow(1.0f - dotIH, 5.0f);

		float sqrtDenom = dotIH + (etaT / etaI) * dotOH;
		float transmissionValue = ((dotOH * dotIH) * D * G * (1.0f - F)) /
			(cosThetaI * cosThetaO * sqrtDenom * sqrtDenom);

		return albedo * transmissionValue * pbr.transmission * cosThetaO;
	}
}

__device__ glm::vec3 Material::samplePbrGGXBrdf(const glm::vec3& normal, const glm::vec3& wi, const glm::vec2& random, bool bInside, float& outPdf)
{
	float transmissionFactor = pbr.transmission;

	float etaI = bInside ? ior : 1.0f;
	float etaT = bInside ? 1.0f : ior;
	float etaRatio = etaI / etaT;

	float r0 = (etaI - etaT) / (etaI + etaT);
	r0 = r0 * r0;
	float cosThetaI = glm::max(0.0f, glm::dot(normal, wi));
	float F = r0 + (1.0f - r0) * glm::pow(1.0f - cosThetaI, 5.0f);

	float opaqueProbability = 1.0f - transmissionFactor;

	if (random.x < opaqueProbability) {
		// Reflection
		glm::vec2 remappedRandom(random.x / opaqueProbability, random.y);

		float specularProbability = glm::mix(0.5f, 1.0f, pbr.metallic);
		if (remappedRandom.x > specularProbability) {
			glm::vec2 diffuseRandom((remappedRandom.x - specularProbability) / (1.0f - specularProbability), remappedRandom.y);
			glm::vec3 wo = Samplers::sampleCosineWeightedHemisphere(normal, diffuseRandom, outPdf);
			outPdf *= (1.0f - specularProbability) * opaqueProbability;
			return wo;
		}
		else {
			glm::vec2 specularRandom(remappedRandom.x / specularProbability, remappedRandom.y);
			glm::vec3 wo = Samplers::sampleWorldGGX(normal, wi, specularRandom, pbr.roughness, outPdf);
			outPdf *= specularProbability * opaqueProbability;
			return wo;
		}
	}
	else {
		// Refraction

	}
}
