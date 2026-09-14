#pragma once

#include <glm/glm.hpp>

struct Ray {
	glm::vec3 origin;
	glm::vec3 direction;
	int pixelIndex;
};

struct PathState {
	glm::vec3 throughput;
	glm::vec3 accumulatedColor;
	int bounceCount;
	bool active;
};