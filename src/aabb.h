#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <algorithm>
#include <limits>

#include "pathTraceCommon.h"

struct AABB {
	glm::vec3 min = glm::vec3((std::numeric_limits<float>::infinity)());
	glm::vec3 max = glm::vec3(-(std::numeric_limits<float>::infinity)());

	void expand(const glm::vec3& point) {
		min = glm::min(min, point);
		max = glm::max(max, point);
	}

	void expand(const AABB& other) {
		min = glm::min(min, other.min);
		max = glm::max(max, other.max);
	}

	static AABB fromTriangle(const Triangle& triangle) {
		AABB box;
		box.expand(triangle.v0);
		box.expand(triangle.v1);
		box.expand(triangle.v2);
		return box;
	}
};