#pragma once

#include <glm/glm.hpp>

#include "pathTraceCommon.h"
#include "microfacet.h"

enum class MaterialType : int {
    OpaqueDiffuse = 0, // Simple diffuse material
	PerfectSpecular = 1, // Mirror or glass material. Uses blinnPhong.bRefractive to distinguish
	BlinnPhong = 2, // Glossy specular material using Blinn-Phong model
	PbrMetallicRoughness = 3, // Physically based rendering material using metallic-roughness workflow
};

struct Material
{
	MaterialType type = MaterialType::OpaqueDiffuse;
	glm::vec3 albedo = glm::vec3(1.0, 0.0, 0.203);
    float emittance = 0.0f;
    float ior = 1.0f;

    union {
        // Obj / json format material type
        struct {
			glm::vec3 specularColor = glm::vec3(1.0, 0.0, 0.203);
            float exponent = 1.0f;
			bool bRefractive = false;
        } blinnPhong;

		// glTF format material type
        struct {
            float metallic = 0.0f;
            float roughness = 1.0f;
            float transmission = 0.0f;
        } pbr;
	};

    __device__ bool isDelta() const {
        if (type == MaterialType::PerfectSpecular) {
            return true;
        }
        if(type == MaterialType::BlinnPhong && blinnPhong.exponent > 1000.0f) {
            return true;
		}
        if (type == MaterialType::PbrMetallicRoughness && pbr.roughness < 0.08f) {
            return true;
        }
        return false;
	}

	__device__ glm::vec3 evaluate(
        const glm::vec3& n, 
        const glm::vec3& wi, 
        const glm::vec3& wo, 
        bool bInside, 
        bool bDirectionGeneratedFromBrdf);

    __device__ float pdf(
        const glm::vec3& n, 
        const glm::vec3& wi, 
        const glm::vec3& wo, 
        bool bInside, 
		bool bDirectionGeneratedFromBrdf);

    __device__ void sample(
        const glm::vec3& n, 
        const glm::vec3& wi, 
        const glm::vec3& random, 
        bool bInside, 
        glm::vec3& wo, 
        glm::vec3& outThroughput, 
        float& outPdf, 
		bool& bIsTransmission);

    __device__ glm::vec3 sampleBrdf(
        const glm::vec3& normal,
        const glm::vec3& wi,
        const glm::vec2& random,
        bool bInside,
        float& outPdf);

    __device__ float getBrdfPdf(
        const glm::vec3& normal,
        const glm::vec3& wi,
        const glm::vec3& wo,
		bool bDirectionGeneratedFromBrdf);

    __device__ glm::vec3 evaluateBrdf(
        const glm::vec3& normal,
        const glm::vec3& wi,
        const glm::vec3& wo,
        bool bDirectionGeneratedFromBrdf
    );

    __device__ MicrofacetScattering getSpecularScattering() const {
		MicrofacetScattering scattering{};
        scattering.ior = ior;

        if(type == MaterialType::BlinnPhong) {
            scattering.model = DistributionModel::BlinnPhong;
            scattering.specularExponent = blinnPhong.exponent;
            scattering.f0 = blinnPhong.specularColor;
        }
        else if(type == MaterialType::PbrMetallicRoughness) {
            scattering.model = DistributionModel::GGX;
            scattering.alpha = pbr.roughness * pbr.roughness;
            scattering.f0 = albedo;
		}

        return scattering;
    }

private:
	// OBJ/JSON material type BRDFs
    __device__ glm::vec3 evaluateDiffuseBrdf(
        const glm::vec3& normal,
        const glm::vec3& outgoingDirection);

    __device__ glm::vec3 sampleDiffuseBrdf(
        const glm::vec3& normal,
        const glm::vec2& random,
        float& outPdf);

    __device__ glm::vec3 evaluateGlossySpecularBrdf(
        const glm::vec3& normal,
        const glm::vec3& wi,
        const glm::vec3& outgoingDirection);

    __device__ glm::vec3 sampleGlossySpecularBrdf(
        const glm::vec3& normal,
        const glm::vec3& wi,
        const glm::vec2& random,
        float& outPdf);

    __device__ glm::vec3 evaluatePerfectSpecularBrdf(
        bool bDirectionGeneratedFromBrdf
    );

    __device__ glm::vec3 samplePerfectSpecularBrdf(
        const glm::vec3& normal,
        const glm::vec3& wi,
        float random,
        bool bInside,
        float& outPdf
    );

	// glTF material type BRDFs
    __device__ glm::vec3 evaluatePbrGGXBrdf(
        const glm::vec3& normal,
        const glm::vec3& wi,
		const glm::vec3& wo,
        bool bInside);

    __device__ glm::vec3 samplePbrGGXBrdf(
        const glm::vec3& normal,
        const glm::vec3& wi,
		const glm::vec2& random,
        bool bInside,
		float& outPdf);
};