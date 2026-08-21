cmake_minimum_required(VERSION 3.11)
project(Game)

set(CMAKE_CXX_STANDARD 17)

find_package(Python 3.11 EXACT COMPONENTS Interpreter Development REQUIRED)

set(PYBIND11_FINDPYTHON ON)
set(pybind11_DIR "C:\\Users\\rohan\\AppData\\Local\\Python\\pythoncore-3.14-64\\Lib\\site-packages\\pybind11\\share\\cmake\\pybind11")

find_package(pybind11)

pybind11_add_module(rlenv src/main.cpp)
#add_executable(rlenv src/test.cpp)

include(FetchContent)

FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	0af55ccecd98d4e5a8d1fad7de25ba429d60e863 #refs/tags/1.0.1
)

FetchContent_MakeAvailable(glm)


target_include_directories(rlenv PUBLIC mujo/include/)
add_library(mujoco SHARED IMPORTED)
set_target_properties(mujoco PROPERTIES
    IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/mujo/lib/mujoco.lib"
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/mujo/bin/mujoco.dll"
)

add_subdirectory(glfw)

target_link_libraries(rlenv PRIVATE mujoco glfw glm::glm)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/mujo/bin/mujoco.dll  DESTINATION Debug)
target_include_directories(rlenv PUBLIC glfw/deps)