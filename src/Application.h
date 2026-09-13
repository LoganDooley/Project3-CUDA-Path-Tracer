#pragma once

#include "Window.h"

#include <vulkan/vulkan_raii.hpp>

#include <memory>

class Application {
public:
	Application();
	~Application();

	const Application& operator=(const Application&) = delete;
	Application(const Application&) = delete;

	const Application& operator=(Application&&) = delete;
	Application(Application&&) = delete;

	void run();

private:
	void InitVulkan();
	void InitCUDAVulkanInterop();
	void InitImGui();

	vk::raii::Context m_vkContext;
	Window m_window;

	vk::raii::Instance m_vkInstance = nullptr;
	vk::raii::SurfaceKHR m_vkSurface = nullptr;
	vk::raii::PhysicalDevice m_vkPhysicalDevice = nullptr;
	vk::raii::Device m_vkDevice = nullptr;
	uint32_t m_graphicsQueueFamilyIndex = 0;
	vk::raii::Queue m_vkGraphicsQueue = nullptr;
	vk::SurfaceFormatKHR m_vkSurfaceFormat;
	vk::Extent2D m_vkSwapchainExtent;
	vk::raii::SwapchainKHR m_vkSwapchain = nullptr;
	std::vector<vk::raii::ImageView> m_vkSwapchainImageViews;

	// CUDA-Vulkan Interop 
	vk::raii::Image m_interopImage = nullptr;
	vk::raii::DeviceMemory m_interopImageMemory = nullptr;
	vk::raii::ImageView m_interopImageView = nullptr;
	HANDLE m_interopImageHandle = nullptr;

	vk::raii::DescriptorPool m_imguiDescriptorPool = nullptr;

	// Command pool + buffer for run loop
	vk::raii::CommandPool m_vkCommandPool = nullptr;
	std::vector<vk::raii::CommandBuffer> m_vkCommandBuffers;

	// Synchronization primitives
	std::vector<vk::raii::Semaphore> m_imageAvailableSemaphores;
	std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
	std::vector<vk::raii::Fence> m_inFlightFences;
	size_t m_currentFrameIndex = 0;
};