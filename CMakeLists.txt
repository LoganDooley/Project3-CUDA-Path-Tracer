cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# GLFW Setup
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "")
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "")
add_subdirectory(external/glfw)

# GLM Setup
add_subdirectory(external/glm)

find_package(Vulkan REQUIRED)

file(GLOB_RECURSE SOURCE_FILES
	"src/*.cpp"
	"src/*.cu"
	"src/*.h"
	"src/*.hpp"
)

# ImGui Sources
file(GLOB IMGUI_SOURCES
	external/imgui/*.cpp
	external/imgui/backends/imgui_impl_glfw.cpp
	external/imgui/backends/imgui_impl_vulkan.cpp
)

add_executable(cis565_path_tracer ${SOURCE_FILES} ${IMGUI_SOURCES})

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
	src/
	external/stb/
	external/imgui/
	external/imgui/backends
)

target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE GLFW_INCLUDE_VULKAN)

target_link_libraries(${CMAKE_PROJECT_NAME} 
PRIVATE 
	Vulkan::Vulkan
	glfw
	glm::glm
)

if(WIN32)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE VK_USE_PLATFORM_WIN32_KHR)
elseif(UNIX AND NOT APPLE)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE VK_USE_PLATFORM_XLIB_KHR)
endif()	