#pragma once

#include <glm/glm.hpp>

#include "pathTraceCommon.h"

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

    __device__ bool isGlossy() const {
        return specular.color.x > 0.0f || specular.color.y > 0.0f || specular.color.z > 0.0f;
    }

    __device__ bool isSpecular() const {
        return hasReflective > 0.0f || hasRefractive > 0.0f;
    }

    __device__ glm::vec3 pickOugoingDirection(
        const glm::vec3& normal,
        const glm::vec3& wi,
        const glm::vec2& random,
        bool bInside,
        float& outPdf);

    __device__ glm::vec3 evaluateBrdf(
        const glm::vec3& normal,
        const glm::vec3& wi,
        const glm::vec3& wo,
        bool bDirectionGeneratedFromBrdf
    );

    __device__ glm::vec3 evaluateDiffuseBrdf(
        const glm::vec3& normal,
        const glm::vec3& outgoingDirection);

    __device__ glm::vec3 pickDiffuseOutgoingDirection(
        const glm::vec3& normal,
        const glm::vec2& random,
        float& outPdf);

    __device__ glm::vec3 evaluateGlossySpecularBrdf(
        const glm::vec3& normal,
        const glm::vec3& wi,
        const glm::vec3& outgoingDirection);

    __device__ glm::vec3 pickGlossySpecularOutgoingDirection(
        const glm::vec3& normal,
        const glm::vec3& wi,
        const glm::vec2& random,
        float& outPdf);

    __device__ glm::vec3 evaluatePerfectSpecularBrdf(
        bool bDirectionGeneratedFromBrdf
    );

    __device__ glm::vec3 pickPerfectSpecularOutgoingDirection(
        const glm::vec3& normal,
        const glm::vec3& wi,
        float random,
        bool bInside,
        float& outPdf
    );
};