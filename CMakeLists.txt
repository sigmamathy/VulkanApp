cmake_minimum_required(VERSION 3.10)

project(JustTest)

set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE FILEPATH "vcpkg toolchain file")
find_package(glfw3 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} "src/main.cpp" "src/app.cpp")

target_include_directories(${PROJECT_NAME} PUBLIC $ENV{VULKAN_SDK}/Include)
target_link_directories(${PROJECT_NAME} PUBLIC $ENV{VULKAN_SDK}/Lib)
target_link_libraries(${PROJECT_NAME} glfw vulkan-1)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
add_compile_definitions(SHADERS_BIN_DIR="${PROJECT_SOURCE_DIR}/shaders/bin/")