cmake_minimum_required(VERSION 3.20)
project(BMWCameraCapture LANGUAGES CXX)

set(RESHADE_INCLUDE_DIR "" CACHE PATH "Path to ReShade include directory containing reshade.hpp")

if (NOT EXISTS "${RESHADE_INCLUDE_DIR}/reshade.hpp")
    message(FATAL_ERROR "Set RESHADE_INCLUDE_DIR to the ReShade include directory that contains reshade.hpp")
endif()

add_library(BMWCameraCapture SHARED BMWCameraCapture.cpp)
target_compile_features(BMWCameraCapture PRIVATE cxx_std_17)
target_include_directories(BMWCameraCapture PRIVATE "${RESHADE_INCLUDE_DIR}")
target_compile_definitions(BMWCameraCapture PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

set_target_properties(BMWCameraCapture PROPERTIES
    OUTPUT_NAME "BMWCameraCapture"
    SUFFIX ".addon64"
)

add_library(GTACameraCapture SHARED GTACameraCapture.cpp)
target_compile_features(GTACameraCapture PRIVATE cxx_std_17)
target_include_directories(GTACameraCapture PRIVATE "${RESHADE_INCLUDE_DIR}")
target_compile_definitions(GTACameraCapture PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)

set_target_properties(GTACameraCapture PROPERTIES
    OUTPUT_NAME "GTACameraCapture"
    SUFFIX ".addon64"
)
