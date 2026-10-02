#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "mathHelpers.h"

enum class DistributionModel {
	BlinnPhong = 0,
	GGX = 1
};

namespace Microfacet {
	// Normal Distribution functions
	__device__ inline float D_BlinnPhong(float dotNH, float specularExponent) {
		if (dotNH <= 0.0f) {
			return 0.0f;
		}

		return ((specularExponent + 2.0f) / (2.0f * glm::pi<float>())) * powf(dotNH, specularExponent);
	}

	__device__ inline float D_GGX(float dotNH, float alpha) {
		if (dotNH <= 0.0f) {
			return 0.0f;
		}

		float alpha2 = alpha * alpha;
		float denom = dotNH * dotNH * (alpha2 - 1.0f) + 1.0f;
		return alpha2 / (glm::pi<float>() * denom * denom);
	}

	// PDF functions
	__device__ inline float PDF_BlinnPhong(float dotNH, float specularExponent) {
		return D_BlinnPhong(dotNH, specularExponent) * dotNH;
	}

	__device__ inline float PDF_GGX(float dotNH, float alpha) {
		return D_GGX(dotNH, alpha) * dotNH;
	}

	// Geometry functions
	__device__ inline float G_VCavity(float dotNH, float dotNV, float cosThetaO, float dotLH) {
		if(dotLH <= 0.0f) {
			return 0.0f;
		}
		return glm::min(1.0f, glm::min(
			(2.0f * dotNH * dotNV) / dotLH, 
			(2.0f * dotNH * cosThetaO) / dotLH)
		);
	}

	__device__ inline float G1_SmithGGX(float dotNX, float alpha) {
		if (dotNX <= 0.0f) {
			return 0.0f;
		}

		float alpha2 = alpha * alpha;
		float dotNX2 = dotNX * dotNX;
		return 2.0f * dotNX / (dotNX + glm::sqrt(alpha2 + (1.0f - alpha2) * dotNX2));
	}

	__device__ inline float G_SmithGGX(float dotNV, float cosThetaO, float alpha) {
		return G1_SmithGGX(dotNV, alpha) * G1_SmithGGX(cosThetaO, alpha);
	}

	// Fresnel
	__device__ inline glm::vec3 F_Schlick(float dotLH, const glm::vec3& f0) {
		return f0 + (glm::vec3(1.0f) - f0) * glm::pow(1.0f - glm::max(0.0f, dotLH), 5.0f);
	}

	// Samplers
	__device__ inline glm::vec3 Sample_BlinnPhong(const glm::vec2& random, float specularExponent) {
		float cosTheta = glm::pow(random.x, 1.0f / (specularExponent + 1.0f));
		float sinTheta = glm::sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
		float phi = 2.0f * glm::pi<float>() * random.y;

		return glm::vec3(
			sinTheta * glm::cos(phi),
			sinTheta * glm::sin(phi),
			cosTheta
		);
	}

	__device__ inline glm::vec3 Sample_GGX(const glm::vec2& random, float alpha) {
		float alpha2 = alpha * alpha;
		float cosTheta = glm::sqrt(glm::max(0.0f, (1.0f - random.x) / (random.x * (alpha2 - 1.0f) + 1.0f)));
		float sinTheta = glm::sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));
		float phi = 2.0f * glm::pi<float>() * random.y;
		return glm::vec3(
			sinTheta * glm::cos(phi), 
			sinTheta * glm::sin(phi), 
			cosTheta
		);
	}
};

struct MicrofacetScattering {
	DistributionModel model;

	union
	{
		float specularExponent; // For Blinn-Phong
		float alpha; // For GGX
	};

	glm::vec3 f0;
	float ior;

	__device__ glm::vec3 evaluate(const glm::vec3& n, const glm::vec3& wi, const glm::vec3& wo, bool bInside) const {
		float cosThetaI = glm::max(0.0f, glm::dot(n, wi));
		float cosThetaO = glm::dot(n, wo);

		if (cosThetaI <= 0.0f) {
			return glm::vec3(0.0f);
		}

		bool bIsReflection = cosThetaO > 0.0f;

		float absCosThetaO = glm::abs(cosThetaO);

		if (absCosThetaO <= 0.0f) {
			return glm::vec3(0.0f);
		}

		float etaI = bInside ? ior : 1.0f;
		float etaT = bInside ? 1.0f : ior;

		if (bIsReflection) {
			glm::vec3 H = glm::normalize(wi + wo);
			float dotNH = glm::max(0.0f, glm::dot(n, H));
			float dotHO = glm::max(0.0f, glm::dot(H, wo));

			float D = (model == DistributionModel::BlinnPhong)
				? Microfacet::D_BlinnPhong(dotNH, specularExponent) 
				: Microfacet::D_GGX(dotNH, alpha);

			float G = (model == DistributionModel::BlinnPhong)
				? Microfacet::G_VCavity(dotNH, cosThetaI, absCosThetaO, dotHO) 
				: Microfacet::G_SmithGGX(cosThetaI, absCosThetaO, alpha);

			glm::vec3 F = Microfacet::F_Schlick(dotHO, f0);

			return (D * G * F) / (4.0f * cosThetaI * absCosThetaO);
		}
		else {
			glm::vec3 H = glm::normalize(wi + wo * (etaT / etaI));
			if (glm::dot(n, H) < 0.0f) {
				H = -H;
			}

			float dotNH = glm::max(0.0f, glm::dot(n, H));
			float dotIH = glm::max(0.0f, glm::dot(wi, H));
			float dotOH = glm::max(0.0f, glm::dot(wo, H));

			float D = (model == DistributionModel::BlinnPhong)
				? Microfacet::D_BlinnPhong(dotNH, specularExponent) 
				: Microfacet::D_GGX(dotNH, alpha);

			float G = (model == DistributionModel::BlinnPhong)
				? Microfacet::G_VCavity(dotNH, cosThetaI, absCosThetaO, dotOH)
				: Microfacet::G_SmithGGX(cosThetaI, absCosThetaO, alpha);

			float r0 = (etaI - etaT) / (etaI + etaT);
			r0 = r0 * r0;
			float F = r0 + (1.0f - r0) * glm::pow(1.0f - dotIH, 5.0f);

			float sqrtDenom = dotIH + (etaT / etaI) * dotOH;
			if(glm::abs(sqrtDenom) < 1e-6f) {
				return glm::vec3(0.0f);
			}
			
			float transmissionValue = ((dotOH * dotIH) * D * G * (1.0f - F)) /
				(cosThetaI * absCosThetaO * sqrtDenom * sqrtDenom);

			return f0 * transmissionValue;
		}
	}

	__device__ float pdf(const glm::vec3& n, const glm::vec3& wi, const glm::vec3& wo, bool bInside) const {
		float cosThetaI = glm::dot(n, wi);
		float cosThetaO = glm::dot(n, wo);
		if (cosThetaI <= 0.0f) {
			return 0.0f;
		}

		bool bIsReflection = cosThetaO > 0.0f;
		float absCosThetaO = glm::abs(cosThetaO);
		if (absCosThetaO <= 0.0f) {
			return 0.0f;
		}

		float etaI = bInside ? ior : 1.0f;
		float etaT = bInside ? 1.0f : ior;

		glm::vec3 H;
		if (bIsReflection) {
			H = glm::normalize(wi + wo);
		}
		else {
			H = glm::normalize(wi + wo * (etaT / etaI));
			if (glm::dot(n, H) < 0.0f) H = -H;
		}

		float dotNH = glm::max(0.0f, glm::dot(n, H));
		float dotIH = glm::max(0.0f, glm::dot(wi, H));
		float dotOH = glm::max(0.0f, glm::dot(wo, H));

		float pdfHalf = (model == DistributionModel::BlinnPhong)
			? Microfacet::PDF_BlinnPhong(dotNH, specularExponent)
			: Microfacet::PDF_GGX(dotNH, alpha);

		float r0 = (etaI - etaT) / (etaI + etaT);
		r0 = r0 * r0;
		float F = r0 + (1.0f - r0) * glm::pow(1.0f - dotIH, 5.0f);

		float reflectionProb = glm::clamp(F, 0.10f, 0.90f);
		if(model == DistributionModel::BlinnPhong) {
			reflectionProb = 1.0f;
		}

		if (bIsReflection) {
			return (pdfHalf / (4.0f * dotIH)) * reflectionProb;
		}
		else {
			float sqrtDenom = dotIH + (etaT / etaI) * dotOH;
			if (glm::abs(sqrtDenom) < 1e-6f) {
				return 0.0f;
			}

			float transmissionProb = 1.0f - reflectionProb;
			return (pdfHalf * dotOH / (sqrtDenom * sqrtDenom)) * transmissionProb;
			return (pdfHalf * (etaT * etaT / (etaI * etaI)) * dotOH / (sqrtDenom * sqrtDenom)) * transmissionProb;
		}
	}

	__device__ void sample(
		const glm::vec3& n, const glm::vec3& wi,
		const glm::vec3& random, bool bInside,
		glm::vec3& out_wo, glm::vec3& out_throughput, float& out_pdf, bool& out_isTransmission) const
	{
		float cosThetaI = glm::dot(n, wi);
		if (cosThetaI <= 0.0f) {
			out_pdf = 0.0f;
			out_throughput = glm::vec3(0.0f);
			return;
		}

		
		glm::vec3 H_local = (model == DistributionModel::GGX)
			? Microfacet::Sample_GGX(glm::vec2(random), alpha)
			: Microfacet::Sample_BlinnPhong(glm::vec2(random), specularExponent);

		// Transform to world space
		glm::vec3 tangent, bitangent;
		MathHelpers::createCoordinateSystem(n, tangent, bitangent);
		glm::vec3 H = glm::normalize(tangent * H_local.x + bitangent * H_local.y + n * H_local.z);

		float dotIH = glm::max(0.0f, glm::dot(wi, H));
		float etaI = bInside ? ior : 1.0f;
		float etaT = bInside ? 1.0f : ior;

		float r0 = (etaI - etaT) / (etaI + etaT);
		r0 = r0 * r0;
		float F = r0 + (1.0f - r0) * glm::pow(1.0f - dotIH, 5.0f);
		float reflectionProb = glm::clamp(F, 0.10f, 0.90f);
		if(model == DistributionModel::BlinnPhong) {
			reflectionProb = 1.0f;
		}

		float dotNH = glm::max(0.0f, glm::dot(n, H));
		float pdfHalf = (model == DistributionModel::GGX) 
			? Microfacet::PDF_GGX(dotNH, alpha) 
			: Microfacet::PDF_BlinnPhong(dotNH, specularExponent);

		if (random.z < reflectionProb) {
			// Reflect
			out_isTransmission = false;
			out_wo = glm::reflect(-wi, H);

			float cosThetaO = glm::dot(n, out_wo);
			if (cosThetaO <= 0.0f) { 
				out_pdf = 0.0f; 
				out_throughput = glm::vec3(0.0f); 
				return; 
			}

			glm::vec3 f = evaluate(n, wi, out_wo, bInside);
			out_pdf = (pdfHalf / (4.0f * dotIH)) * reflectionProb;
			out_throughput = (out_pdf <= 0.0f) ?
				glm::vec3(0.0f) :
				(f * cosThetaO) / out_pdf;
		}
		else {
			// Refract
			float eta = etaI / etaT;
			glm::vec3 refract_vector = glm::refract(-wi, H, eta);

			// Total internal reflection
			if (glm::dot(refract_vector, refract_vector) < 0.0001f) {
				out_isTransmission = false;
				out_wo = glm::reflect(-wi, H);

				float cosThetaO = glm::dot(n, out_wo);
				if (cosThetaO <= 0.0f) { 
					out_pdf = 0.0f; 
					out_throughput = glm::vec3(0.0f); 
					return; 
				}

				glm::vec3 f = evaluate(n, wi, out_wo, bInside);

				float transmissionProb = 1.0f - reflectionProb;
				out_pdf = (pdfHalf / (4.0f * dotIH)) * transmissionProb;
				out_throughput = (out_pdf <= 0.0f) 
					? glm::vec3(0.0f) 
					: (f * cosThetaO) / out_pdf;
				return;
			}

			out_isTransmission = true;
			out_wo = glm::normalize(refract_vector);

			float cosThetaO = glm::dot(n, out_wo);
			float absCosThetaO = glm::abs(cosThetaO);
			float dotOH = glm::max(0.0f, glm::abs(glm::dot(out_wo, H)));


			float sqrtDenom = dotIH + (etaT / etaI) * dotOH;
			if (glm::abs(sqrtDenom) < 1e-6f) { 
				out_pdf = 0.0f; 
				out_throughput = glm::vec3(0.0f); 
				return; 
			}

			glm::vec3 f = evaluate(n, wi, out_wo, bInside);
			float transmissionProb = 1.0f - reflectionProb;

			out_pdf = (pdfHalf * (etaT * etaT / (etaI * etaI)) * dotOH / (sqrtDenom * sqrtDenom)) * transmissionProb;
			out_throughput = (out_pdf <= 0.0f) 
				? glm::vec3(0.0f) 
				: (f * absCosThetaO) / out_pdf;
		}
	}
};