cmake_minimum_required(VERSION 3.20.0)

project(AI-Engine)


message(STATUS "BUILD_PLATFORM: ${BUILD_PLATFORM}")
string(TOUPPER ${BUILD_PLATFORM} def)
add_definitions("-DBUILD_PLATFORM_${def}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

set(JSON_BuildTests OFF CACHE INTERNAL "")
add_subdirectory(${CMAKE_SOURCE_DIR}/3rdparty/nlohmann_json-3.11)

set(Eigen3_DIR ${CMAKE_SOURCE_DIR}/3rdparty/eigen-3.4.0/share/eigen3/cmake)
find_package(Eigen3 REQUIRED)

set(GLOBAL_LINK_3RD_PARTY_LIBS
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)

if(BUILD_PLATFORM STREQUAL "linux")
    find_package(OpenMP REQUIRED)

    # opencv
    # set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-mobile-4.10.0-ubuntu-2204/lib/cmake/opencv4)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-static-full) # for debug build
    
    # onnxruntime
    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/lib/libonnxruntime.a")

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

    find_package(OpenMP REQUIRED)

elseif(BUILD_PLATFORM STREQUAL "macos")
    

    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON) #Set this to ON if you want to use GNU++
    set(CMAKE_CXX_STANDARD 17)

    find_package(OpenMP REQUIRED)

    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-mobile-4.10.0-macos)

    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2/lib/libonnxruntime.a")

elseif(BUILD_PLATFORM STREQUAL "ios")
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON) #Set this to ON if you want to use GNU++
    set(CMAKE_CXX_STANDARD 17)
elseif(BUILD_PLATFORM STREQUAL "windows")

endif()

find_package(OpenCV REQUIRED)


add_subdirectory(tools)
add_subdirectory(examples)
# add_subdirectory(src)


message("")
message("Compile Summary:")
message("========================================================================")
# ============================ Options ===========================
message("")

message(STATUS "  Build options: ")
message(STATUS "    BUILD_PLATFORM:"           ${BUILD_PLATFORM}      )
message(STATUS "    ANDROID_ABI:"              ${ANDROID_ABI}         )
# ============================ OpenCV ===========================
message("")
message(STATUS "  OpenCV: ")
message(STATUS "    Version:" "${OpenCV_VERSION}" )
message(STATUS "    OpenCV_DIR:" "${OpenCV_DIR}"  )
# message(STATUS "    OpenCV_LIBS:" "${OpenCV_LIBS}" )
# message(STATUS "    OpenCV_INCLUDE_DIRS:" "${OpenCV_INCLUDE_DIRS}" )

message("")
message(STATUS "  OpenMP: ")
message(STATUS "    OpenMP_FOUND:" "${OpenMP_FOUND}"  )

message("")
message(STATUS "  SOURCE: ")
message(STATUS "    SOURCES:" "${SOURCES}" )
message("========================================================================")