cmake_minimum_required(VERSION 3.20.0)

# project(GustoEngine)


message(STATUS "BUILD_PLATFORM: ${BUILD_PLATFORM}")
string(TOUPPER ${BUILD_PLATFORM} def)
add_definitions("-DBUILD_PLATFORM_${def}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(SOURCES
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
)

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/nlohmann_json-3.11)

set(Eigen3_DIR ${CMAKE_SOURCE_DIR}/3rdparty/eigen-3.4.0/share/eigen3/cmake)
find_package(Eigen3 REQUIRED)

if(BUILD_PLATFORM STREQUAL "linux")
elseif(BUILD_PLATFORM STREQUAL "android")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g0")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    if (ANDROID_ABI STREQUAL "arm64-v8a")
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/aarch64/libomp.a")
    elseif (ANDROID_ABI STREQUAL "armeabi-v7a")
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/arm/libomp.a")
    elseif(ANDROID_ABI STREQUAL "x86")
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/i386/libomp.a")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/x86_64/libomp.a")
    endif()
elseif(BUILD_PLATFORM STREQUAL "macos")
    set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/llvm/19.1.3/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm/19.1.3/bin/clang++")
    set(OPENMP_LIBRARIES "/opt/homebrew/Cellar/llvm/19.1.3/lib")
    set(OPENMP_INCLUDES "/opt/homebrew/Cellar/llvm/19.1.3/include")

    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON) #Set this to ON if you want to use GNU++
    set(CMAKE_CXX_STANDARD 17)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
    endif()
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.a")
elseif(BUILD_PLATFORM STREQUAL "ios")
    # set(OPENMP_LIBRARIES "/opt/homebrew/Cellar/llvm/19.1.3/lib")
    # set(OPENMP_INCLUDES "/opt/homebrew/Cellar/llvm/19.1.3/include")
    # enable_language(CXX)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON) #Set this to ON if you want to use GNU++
    set(CMAKE_CXX_STANDARD 17)
elseif(BUILD_PLATFORM STREQUAL "windows")


endif()



add_subdirectory(tools)
# add_subdirectory(examples)
# add_subdirectory(src)
