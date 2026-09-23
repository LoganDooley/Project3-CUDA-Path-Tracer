#include "devScene.h"

#include "samplers.h"

__device__ IntersectionData DevScene::intersect(const Ray& ray) {
    IntersectionData closestIntersection = IntersectionData{};
    for (int i = 0; i < m_geometryCount; i++) {
        IntersectionData intersection = IntersectionStatics::intersectGeometry(ray, dev_geometry[i]);
        if (intersection.t > 0.0f) {
            if (closestIntersection.t < 0.0f || intersection.t < closestIntersection.t) {
                closestIntersection = intersection;
            }
        }
    }

    return closestIntersection;
}

__device__ bool DevScene::isVisible(const Ray& ray, float tMax)
{
    IntersectionData result = intersect(ray);
    return result.t < 0.0f || result.t > tMax;
}

__device__ glm::vec3 DevScene::nextEventEsimation(const glm::vec4& random, 
    const Ray& incomingRay, 
    const IntersectionData& intersectionData)
{
    if (m_lightCount == 0) {
        return glm::vec3(0.0f);
    }

    float lightIndexPdf = 1.0 / (float)m_lightCount;

    int chosenLightIndex = (int)(random.w * (float)m_lightCount);

    if (chosenLightIndex >= m_geometryCount) {
        return glm::vec3(0.0f);
    }

    Geom lightGeometry = dev_geometry[chosenLightIndex];

    if (lightGeometry.materialid >= m_materialCount) {
        return glm::vec3(0.0f);
    }

    Material lightMaterial = dev_materials[lightGeometry.materialid];
    if (lightMaterial.emittance <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 lightNormal = glm::vec3(0.0f);
    float lightSurfacePdf = 1.0f;
    glm::vec3 lightPosition = Samplers::sampleGeometry(lightGeometry, glm::vec3(random), lightNormal, lightSurfacePdf);

    if (lightSurfacePdf <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 hitPoint = incomingRay.getPositionAtTime(intersectionData.t);

    Ray visibilityRay;
    const float epsilon = 0.0001f;
    visibilityRay.origin = hitPoint + epsilon * intersectionData.normal;
    
    glm::vec3 rayDirection = lightPosition - visibilityRay.origin;
    float distance = glm::length(rayDirection);
    visibilityRay.direction = rayDirection / distance;

    float cosThetaLight = glm::dot(lightNormal, -visibilityRay.direction);
    float cosThetaSurface = glm::dot(intersectionData.normal, visibilityRay.direction);

    if (cosThetaLight <= 0.0f || cosThetaSurface <= 0.0f) {
        return glm::vec3(0.0f);
    }

    bool bVisible = isVisible(visibilityRay, distance - epsilon);

    if (!bVisible) {
        return glm::vec3(0.0f);
    }

    glm::vec3 emission = lightMaterial.emittance * lightMaterial.color;

    Material surfaceMaterial = dev_materials[intersectionData.materialIndex];

    glm::vec3 brdf = surfaceMaterial.color / glm::pi<float>();

    float geometryTerm = cosThetaLight / (distance * distance);

    return (brdf * emission * cosThetaSurface * geometryTerm) / (lightSurfacePdf * lightIndexPdf);
}
