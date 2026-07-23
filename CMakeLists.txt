cmake_minimum_required(VERSION 3.20)

project(AIEngine VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(BUILD_PLATFORM "" CACHE STRING "Target platform: linux, android, macos, ios, windows, or wasm")
set_property(CACHE BUILD_PLATFORM PROPERTY STRINGS linux android macos ios windows wasm)

if(NOT BUILD_PLATFORM)
    if(EMSCRIPTEN)
        set(BUILD_PLATFORM wasm)
    elseif(ANDROID)
        set(BUILD_PLATFORM android)
    elseif(WIN32)
        set(BUILD_PLATFORM windows)
    elseif(APPLE)
        set(BUILD_PLATFORM macos)
    elseif(UNIX)
        set(BUILD_PLATFORM linux)
    endif()
endif()

set(AI_ENGINE_SUPPORTED_PLATFORMS linux android macos ios windows wasm)
list(FIND AI_ENGINE_SUPPORTED_PLATFORMS "${BUILD_PLATFORM}" AI_ENGINE_PLATFORM_INDEX)
if(AI_ENGINE_PLATFORM_INDEX EQUAL -1)
    message(FATAL_ERROR
        "Unsupported BUILD_PLATFORM='${BUILD_PLATFORM}'. "
        "Choose one of: ${AI_ENGINE_SUPPORTED_PLATFORMS}")
endif()

string(TOUPPER "${BUILD_PLATFORM}" AI_ENGINE_PLATFORM_DEFINE)
add_compile_definitions("BUILD_PLATFORM_${AI_ENGINE_PLATFORM_DEFINE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(AI_ENGINE_BUILD_CORE "Build the native inference library" ON)
option(AI_ENGINE_BUILD_UNITY "Build the Unity C API library" ON)
option(AI_ENGINE_BUILD_EXAMPLES "Build example executables" OFF)
option(AI_ENGINE_BUILD_STANDALONE_TOOLS "Build standalone post-processing libraries" OFF)
option(AI_ENGINE_BUILD_6D_TRACKING "Build the experimental 6D tracking tools" OFF)

if(BUILD_PLATFORM STREQUAL "wasm")
    set(AI_ENGINE_BUILD_CORE OFF CACHE BOOL "" FORCE)
    set(AI_ENGINE_BUILD_UNITY OFF CACHE BOOL "" FORCE)
endif()

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(3rdparty/nlohmann_json-3.11)

set(Eigen3_DIR "${CMAKE_SOURCE_DIR}/3rdparty/eigen-3.4.0/share/eigen3/cmake")
find_package(Eigen3 REQUIRED CONFIG)

include(cmake/Dependencies.cmake)

set(SOURCES
    "${CMAKE_SOURCE_DIR}/src/utils.cc"
    "${CMAKE_SOURCE_DIR}/src/BaseONNX.cc"
)

set(DETECTION_2D_LIBS
    "${CMAKE_SOURCE_DIR}/src/detector2d_family.cc"
)

set(FACE_MODEL_LIBS
    "${CMAKE_SOURCE_DIR}/src/face_model_family.cc"
)

set(HUMAN_SEG_LIBS
    "${CMAKE_SOURCE_DIR}/src/human_seg_family.cc"
)

set(HUMAN_POSE_LIBS
    "${CMAKE_SOURCE_DIR}/src/human_pose_family.cc"
    "${CMAKE_SOURCE_DIR}/src/two_stage_human_pose_extractor_2d.cc"
)

set(NMS_LIBS
    "${CMAKE_SOURCE_DIR}/tools/nms/nms.cc"
)

set(FACE_GEOMETRY_LIBS
    "${CMAKE_SOURCE_DIR}/tools/face_geometry/face_geometry.cc"
    "${CMAKE_SOURCE_DIR}/tools/face_geometry/procrustes_solver.cc"
    "${CMAKE_SOURCE_DIR}/tools/face_geometry/geometry_pipeline.cc"
    "${CMAKE_SOURCE_DIR}/tools/face_geometry/calculator.cc"
    "${CMAKE_SOURCE_DIR}/src/multi_stage_face_geometry_3d.cc"
)

set(AI_ENGINE_MODULES
    ${SOURCES}
    ${FACE_MODEL_LIBS}
    ${HUMAN_SEG_LIBS}
    ${HUMAN_POSE_LIBS}
    ${DETECTION_2D_LIBS}
    ${FACE_GEOMETRY_LIBS}
    ${NMS_LIBS}
)

if(AI_ENGINE_BUILD_CORE)
    add_subdirectory(src)
endif()

if(AI_ENGINE_BUILD_UNITY)
    add_subdirectory(src_unity_api)
endif()

if(BUILD_PLATFORM STREQUAL "wasm")
    add_subdirectory(src_wasm_api)
endif()

if(AI_ENGINE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(AI_ENGINE_BUILD_STANDALONE_TOOLS OR AI_ENGINE_BUILD_6D_TRACKING)
    add_subdirectory(tools)
endif()

message(STATUS "")
message(STATUS "AI-Engine configuration")
message(STATUS "  Platform: ${BUILD_PLATFORM}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Core library: ${AI_ENGINE_BUILD_CORE}")
message(STATUS "  Unity bindings: ${AI_ENGINE_BUILD_UNITY}")
message(STATUS "  Examples: ${AI_ENGINE_BUILD_EXAMPLES}")
message(STATUS "  Standalone tools: ${AI_ENGINE_BUILD_STANDALONE_TOOLS}")
message(STATUS "  Experimental 6D tracking: ${AI_ENGINE_BUILD_6D_TRACKING}")
message(STATUS "  OpenCV: ${OpenCV_VERSION} (${OpenCV_DIR})")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
