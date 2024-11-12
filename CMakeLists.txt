cmake_minimum_required(VERSION 3.20.0)

# project(GustoEngine)


message(STATUS "BUILD_PLATFORM: ${BUILD_PLATFORM}")
string(TOUPPER ${BUILD_PLATFORM} def)
add_definitions("-DBUILD_PLATFORM_${def}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")


if(BUILD_PLATFORM STREQUAL "android")
    set(Eigen3_DIR "/usr/share/eigen3/cmake")
endif()

set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/nlohmann_json-3.11)


add_subdirectory(tools)
add_subdirectory(examples)
# add_subdirectory(src)
