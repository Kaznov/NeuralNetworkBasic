
cmake_minimum_required(VERSION 3.5)

set(BUILD_TARGET "NeuralNetworkBasic")

project(${BUILD_TARGET})


if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/include")
  message(FATAL_ERROR "The glfw submodule directory is missing! "
    "You probably did not clone submodules. It is possible to recover "
    "by running \"git submodule update --init --recursive\" on top-level directory")
endif()

set(CMAKE_CXX_STANDARD            17)
set(CMAKE_CXX_STANDARD_REQUIRED   YES)


find_package(OpenGL REQUIRED)
# OpenGL
include_directories(${OPENGL_INCLUDE_DIR})


# [glfw]
# local glad
include_directories(${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/deps")
list(APPEND UI_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/deps/glad/src/glad.c)
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "" FORCE)
set(GLFW_INSTALL OFF CACHE INTERNAL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/glfw/include)
list(APPEND EXT_LIBRARIES glfw)

# [ImGUI] and [imgui-node-editor]
# Use imgui-node-editor's custom imgui
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui-file)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/backends)

# This is required for imgui_impl_opengl3
add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)
list(APPEND UI_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/imgui.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/imgui_widgets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/imgui_tables.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/imgui_draw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/imgui_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/backends/imgui_impl_glfw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/backends/imgui_impl_opengl3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui-file/ImGuiFileBrowser.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui/examples/example_glfw_opengl3/main.cpp
    )


set(NNBASIC_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/DataPoint.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NeuralNetwork.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NeuralNetwork.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NNAliases.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NNLayer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NNLossFun.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NNMomentum.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NNTeacher.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/NNTerminator.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gui/main.cpp
  )

# Increase warning level for clang.
# Only apply source files of `nnview`
# https://stackoverflow.com/questions/13638408/override-compile-flags-for-single-files
IF (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Assume C++ sources
  set_source_files_properties(${NNBASIC_SOURCES} PROPERTIES COMPILE_FLAGS "-Weverything -Wno-system-headers -Werror -Wno-padded -Wno-c++98-compat-pedantic -Wno-documentation -Wno-documentation-unknown-command -Wno-reserved-id-macro")
ENDIF ()


add_executable(${BUILD_TARGET}
    ${NNBASIC_SOURCES}
    ${UI_SOURCES}
)

target_include_directories(
  ${BUILD_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/
)

target_include_directories(${BUILD_TARGET} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/glad/include
)


target_link_libraries(
    ${BUILD_TARGET}
    ${OPENGL_LIBRARIES}
    ${EXT_LIBRARIES}
)

# Install the built executable into (prefix)/bin
install(TARGETS ${BUILD_TARGET} DESTINATION bin)
