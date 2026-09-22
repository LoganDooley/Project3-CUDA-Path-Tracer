#include "devScene.h"

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