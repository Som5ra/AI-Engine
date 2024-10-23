cmake_minimum_required(VERSION 3.20.0)

# project(GustoEngine)


message(STATUS "BUILD_PLATFORM: ${BUILD_PLATFORM}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")


set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
)


add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tools)