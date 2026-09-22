#pragma once

#include <glm/glm.hpp>

#include "pathTraceCommon.h"

class ShadingMaterial {
public:
    __device__ static glm::vec3 evaluateDiffuseMaterial(
        const Material& material,
        const glm::vec3& normal,
        const glm::vec2& random,
        glm::vec3& outOutgoingDirection);

    __device__ static glm::vec3 evaluateGlossySpecularMaterial(
        const Material& material,
        const glm::vec3& normal,
        const glm::vec3& wi,
        const glm::vec2& random,
        glm::vec3& outOutgoingDirection);
};