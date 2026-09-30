#include "devScene.h"

#include "material.h"
#include "samplers.h"

#define USE_TLAS 1

__device__ IntersectionData DevScene::intersect(const Ray& ray) {
    IntersectionData result = IntersectionData{};

#if USE_TLAS
    int nodeStack[32];
    int stackPtr = 0;

    nodeStack[stackPtr++] = 0; // tlas node index 0 is always the root

    while (stackPtr > 0) {
        result.tlasIterationCount++;

        // Pop node off stack
        int nodeIndex = nodeStack[--stackPtr];
        const TLASNode& node = dev_tlasNodes[nodeIndex];

        float tNear;
        if (!node.intersect(ray, tNear)) {
            // Missed this node, skip it
            continue;
        }

        if (result.t > 0.0f && tNear > result.t) {
            // We already have a closer intersection, skip this node
            continue;
        }

        if (node.leftChild == -1) {
            // Hit leaf node, check geometry
			int geomIndex = node.geometryIndex;
            if (geomIndex < 0 || geomIndex >= m_geometryCount) {
                continue;
            }

			Geom geometry = dev_geometry[geomIndex];
			IntersectionData geometryResult = IntersectionStatics::intersectGeometry(ray, geometry, dev_blasNodes, dev_triangles);
            if(geometryResult.t > 0.0f && (result.t < 0.0f || geometryResult.t < result.t)) {
                result = geometryResult;
                result.geometryIndex = geomIndex;
			}
        }
        else {
            // Push children onto stack if there is room
            if (stackPtr + 2 < 32) {
                nodeStack[stackPtr++] = node.leftChild;
                nodeStack[stackPtr++] = node.leftChild + 1;
            }
        }
    }
#else
    for (int i = 0; i < m_geometryCount; i++) {
        IntersectionData intersection = IntersectionStatics::intersectGeometry(ray, dev_geometry[i], dev_blasNodes, dev_triangles);
        if (intersection.t > 0.0f) {
            if (result.t < 0.0f || intersection.t < result.t) {
                result = intersection;
                result.geometryIndex = i;
            }
        }
    }
#endif

    return result;
}

__device__ bool DevScene::isVisible(const Ray& ray, float tMax)
{
    IntersectionData result = intersect(ray);
    return result.t < 0.0f || result.t > tMax;
}

__device__ glm::vec3 DevScene::nextEventEsimation(const glm::vec4& random, 
    const Ray& incomingRay, 
    const IntersectionData& intersectionData,
    float& outPdf)
{
    if (m_lightCount == 0) {
        return glm::vec3(0.0f);
    }

    float lightIndexPdf = 1.0 / (float)m_lightCount;

    int chosenLightIndex = (int)(random.w * (float)m_lightCount);

    if (chosenLightIndex < 0 || chosenLightIndex >= m_lightCount) {
        outPdf = 0.0f;
        return glm::vec3(0.0f);
    }

    Geom lightGeometry = dev_geometry[chosenLightIndex];

    if (lightGeometry.materialid >= m_materialCount) {
        outPdf = 0.0f;
        return glm::vec3(0.0f);
    }

    Material lightMaterial = dev_materials[lightGeometry.materialid];
    if (lightMaterial.emittance <= 0.0f) {
        return glm::vec3(0.0f);
    }

    glm::vec3 lightNormal = glm::vec3(0.0f);
    float lightSurfacePdf = 1.0f;
    glm::vec3 lightPosition = Samplers::sampleGeometry(lightGeometry, dev_triangles, glm::vec3(random), lightNormal, lightSurfacePdf);

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
        outPdf = 0.0f;
        return glm::vec3(0.0f);
    }

    if (!isVisible(visibilityRay, distance - epsilon)) {
        outPdf = 0.0f;
        return glm::vec3(0.0f);
    }

    glm::vec3 emission = lightMaterial.emittance * lightMaterial.color;

    Material surfaceMaterial = dev_materials[intersectionData.materialIndex];

    glm::vec3 brdf = surfaceMaterial.evaluateBrdf(intersectionData.normal, -incomingRay.direction, visibilityRay.direction, false);

    float pArea = lightSurfacePdf;
	float pOmega = pArea * distance * distance / cosThetaLight;
    outPdf = pOmega * lightIndexPdf;

    return (brdf * emission) / outPdf;
}

__device__ float DevScene::getLightPdf(int lightIndex, const glm::vec3& worldPosition, const glm::vec3& worldNormal, float hitTriangleLocalSurfaceArea)
{
    if(lightIndex < 0 || lightIndex >= m_lightCount) {
        return 0.0f;
	}

	Geom lightGeometry = dev_geometry[lightIndex];

    glm::mat3 transform3 = glm::mat3(lightGeometry.transform);
    glm::mat3 invTranspose3 = glm::mat3(lightGeometry.invTranspose);

    float lightIndexPdf = 1.0 / (float)m_lightCount;

    float localPdf = 0.0f;
    glm::vec3 localNormal = glm::vec3(0.0f);

    if (lightGeometry.type == GeomType::SPHERE) {
        constexpr float sphereArea = 4.0f * glm::pi<float>() * 0.25f;
		localPdf = 1.0f / sphereArea;

        localNormal = glm::transpose(transform3) * worldNormal;
        localNormal = glm::normalize(localNormal);
    }
    else if (lightGeometry.type == GeomType::CUBE){
		localPdf = 1.0f / 6.0f;

        glm::vec3 localPos = glm::vec3(lightGeometry.inverseTransform * glm::vec4(worldPosition, 1.0f));

        glm::vec3 absPos = glm::abs(localPos);
        if (absPos.x > absPos.y && absPos.x > absPos.z) {
            localNormal = glm::vec3(glm::sign(localPos.x), 0.0f, 0.0f);
        }
        else if (absPos.y > absPos.z) {
            localNormal = glm::vec3(0.0f, glm::sign(localPos.y), 0.0f);
        }
        else {
            localNormal = glm::vec3(0.0f, 0.0f, glm::sign(localPos.z));
        }
    }
    else if (lightGeometry.type == GeomType::MESH) {
        if(hitTriangleLocalSurfaceArea <= 0.0f) {
            return 0.0f;
		}
        localPdf = 1.0f / (lightGeometry.triangleCount * hitTriangleLocalSurfaceArea);
        localNormal = glm::transpose(transform3) * worldNormal;
        localNormal = glm::normalize(localNormal);
    }
    else {
        return 0.0f;
	}

    glm::vec3 worldNormalScaled = invTranspose3 * localNormal;
    float normalScale = glm::length(worldNormalScaled);

	float det = glm::abs(glm::determinant(transform3));
    float jacobian = det * normalScale;

    if (jacobian <= 0.0f) {
        return 0.0f;
    }

	localPdf /= jacobian;

	return localPdf * lightIndexPdf;
}
