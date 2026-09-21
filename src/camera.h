#pragma once

#include <glm/glm.hpp>

#include "inputState.h"

class Camera {
public:
	Camera() = default;
	Camera(glm::vec3 eye, glm::vec3 lookAt, glm::vec3 up, float fovy);
	~Camera() = default;

	Camera& operator=(const Camera&) = default;
	Camera(const Camera&) = default;

	Camera& operator=(Camera&&) noexcept = default;
	Camera(Camera&&) noexcept = default;

	void tick(float deltaTime, const InputState& inputState);
	void rotate(const glm::vec2& mouseDelta);

	glm::vec3 m_position = glm::vec3(0.0f, 0.0f, 3.0f);
	glm::vec3 m_look = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 m_up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 m_right = glm::vec3(1.0f, 0.0f, 0.0f);
	float m_fovy = 45.0f;
	float m_moveSpeed = 2.0f;
	float m_rotateSpeed = 0.002f;

	bool m_hasMoved = false;
};