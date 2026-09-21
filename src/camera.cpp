#include "camera.h"

#include <glfw/glfw3.h>

#include "mathHelpers.h"

Camera::Camera(glm::vec3 eye, glm::vec3 lookAt, glm::vec3 up, float fovy)
{
	m_position = eye;
	m_look = glm::normalize(lookAt - eye);
	m_up = glm::normalize(up);
	m_right = glm::normalize(glm::cross(m_look, m_up));
	m_fovy = fovy;
}

bool Camera::tick(float deltaTime, const InputState& inputState)
{
	glm::vec3 moveDirection = glm::vec3(0.0f);
	if (inputState.isKeyPressed(GLFW_KEY_W)) {
		moveDirection += m_look;
	}
	if (inputState.isKeyPressed(GLFW_KEY_S)) {
		moveDirection -= m_look;
	}
	if (inputState.isKeyPressed(GLFW_KEY_D)) {
		moveDirection += m_right;
	}
	if (inputState.isKeyPressed(GLFW_KEY_A)) {
		moveDirection -= m_right;
	}
	if (inputState.isKeyPressed(GLFW_KEY_SPACE)) {
		moveDirection += m_up;
	}
	if (inputState.isKeyPressed(GLFW_KEY_LEFT_SHIFT)) {
		moveDirection -= m_up;
	}

	moveDirection = MathHelpers::safeNormalize(moveDirection);

	if (moveDirection == glm::vec3(0.0f)) {
		return false;
	}

	m_position += m_moveSpeed * deltaTime * moveDirection;
	return true;
}
