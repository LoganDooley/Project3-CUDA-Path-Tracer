cmake_minimum_required(VERSION 3.24)
project(cis565_path_tracer LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(Vulkan REQUIRED)

file(GLOB_RECURSE SOURCE_FILES
	"src/*.cpp"
	"src/*.cu"
	"src/*.h"
	"src/*.hpp"
)

add_executable(cis565_path_tracer ${SOURCE_FILES})

target_link_libraries(${CMAKE_PROJECT_NAME} 
PRIVATE 
	Vulkan::Vulkan
)

if(WIN32)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE VK_USE_PLATFORM_WIN32_KHR)
elseif(UNIX AND NOT APPLE)
	target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE VK_USE_PLATFORM_XCB_KHR)
endif()	