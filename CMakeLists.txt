cmake_minimum_required(VERSION 3.20.0)

project(AI-Engine)


message(STATUS "BUILD_PLATFORM: ${BUILD_PLATFORM}")
string(TOUPPER ${BUILD_PLATFORM} def)
add_definitions("-DBUILD_PLATFORM_${def}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")


set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


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
    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/static_lib/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/static_lib/lib/libonnxruntime.a")
    # set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-android-static_lib-1.18.1/headers")
    # set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-build/build/static_lib_xnn/libonnxruntime_static_lib.a")

elseif(BUILD_PLATFORM STREQUAL "android")
    
    set(ANDROID_SYSROOT "${CMAKE_SOURCE_DIR}/android-ndk-r27c-linux/android-ndk-r27c")

    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON) #Set this to ON if you want to use GNU++
    set(CMAKE_CXX_STANDARD 17)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g0 -fPIC")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")

    include_directories("${ANDROID_SYSROOT}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/include")
    
    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-android-static_lib-1.19.2/static_lib_${ANDROID_ABI}/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-android-static_lib-1.19.2/static_lib_${ANDROID_ABI}/lib/libonnxruntime.a")

    # set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-android-static_lib-1.19.2/headers")
    # set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-android-static_lib-1.19.2/${ANDROID_ABI}/libonnxruntime_static_lib.a")

    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-mobile-4.10.0-android-ndk26/${ANDROID_ABI}/install/sdk/native/jni)

    # set(OpenMP_Base_DIR "${CMAKE_SOURCE_DIR}/android-ndk-r26d-linux/android-ndk-r26d/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/17/lib/linux")
    # set(OpenMP_Base_DIR "${CMAKE_SOURCE_DIR}/android-ndk-r27c-linux/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/lib/linux")
    set(OpenMP_Base_DIR "${ANDROID_SYSROOT}/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/lib/linux")

    if (ANDROID_ABI STREQUAL "arm64-v8a")
        set(OpenMP_omp_LIBRARY "${OpenMP_Base_DIR}/aarch64/libomp.a")
    elseif (ANDROID_ABI STREQUAL "armeabi-v7a")
        set(OpenMP_omp_LIBRARY "${OpenMP_Base_DIR}/arm/libomp.a")
    elseif(ANDROID_ABI STREQUAL "x86")
        set(OpenMP_omp_LIBRARY "${OpenMP_Base_DIR}/i386/libomp.a")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(OpenMP_omp_LIBRARY "${OpenMP_Base_DIR}/x86_64/libomp.a")
    endif()

    find_package(OpenMP REQUIRED)

elseif(BUILD_PLATFORM STREQUAL "macos")
    

    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON) #Set this to ON if you want to use GNU++
    set(CMAKE_CXX_STANDARD 17)

    find_package(OpenMP REQUIRED)
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
    
    # set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-mobile-4.10.0-macos/lib/cmake/opencv4)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-4.10.0/build) # for debug build

    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2/lib/libonnxruntime.a")

elseif(BUILD_PLATFORM STREQUAL "ios")
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON) #Set this to ON if you want to use GNU++
    set(CMAKE_CXX_STANDARD 17)

    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-mobile-4.10.0-ios/lib/cmake/opencv4)
    file(GLOB OpenCV_LIB_FILES ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-mobile-4.10.0-ios/lib/*.a)

    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime.xcframework-1.19.2/onnxruntime.xcframework/Headers")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime.xcframework-1.19.2/onnxruntime.xcframework/ios-arm64/libonnxruntime.a")

elseif(BUILD_PLATFORM STREQUAL "windows")
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON) #Set this to ON if you want to use GNU++
    set(CMAKE_CXX_STANDARD 17)
    # set(opencv_perf_test OFF)
    
    # add_compile_definitions("-DBUILD_WITH_STATIC_CRT=ON")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")


    if (MSVC_VERSION GREATER_EQUAL "1900")
        include(CheckCXXCompilerFlag)
        CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
        if (_cpp_latest_flag_supported)
            add_compile_options("/std:c++latest")
        endif()
    endif()
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-mobile-4.10.0/build/install/x64/vc17/staticlib)

    # need msvc compiler
    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-win-x64-static_lib-1.19.2/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-win-x64-static_lib-1.19.2/lib/onnxruntime.lib")

    
    add_definitions(-DNOMINMAX) # Otherwise, std::max() and std::min() won't work
  
    # disable various warnings for MSVC
    # 4244: 'return': conversion from 'unsigned __int64' to 'int', possible loss of data
    # 4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
    # 4305: 'argument': truncation from 'double' to 'const float'
    # 4334: '<<': result of 32-bit shift implicitly converted to 64 bits
    # 4800: 'int': forcing value to bool 'true' or 'false'
    # 4996: 'fopen': This function or variable may be unsafe
    set(disabled_warnings
        /wd4244
        /wd4267
        /wd4305
        /wd4334
        /wd4800
        /wd4996
    )
    message(STATUS "Disabled warnings: ${disabled_warnings}")
    foreach(w IN LISTS disabled_warnings)
    string(APPEND CMAKE_CXX_FLAGS " ${w} ")
    endforeach()

    
    # add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    # add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

endif()

find_package(OpenCV REQUIRED)


include_directories(
    ${CMAKE_SOURCE_DIR}/include
)

set(SOURCES 
    ${CMAKE_SOURCE_DIR}/src/utils.cc
    ${CMAKE_SOURCE_DIR}/src/BaseONNX.cc
)

set(DETECTION_2D_LIBS
    ${CMAKE_SOURCE_DIR}/src/detector2d_family.cc
)

set(FACE_MODEL_LIBS
    ${CMAKE_SOURCE_DIR}/src/face_model_family.cc
)

set(HUMAN_SEG_LIBS
    ${CMAKE_SOURCE_DIR}/src/human_seg_family.cc
)

set(HUMAN_POSE_LIBS
    ${CMAKE_SOURCE_DIR}/src/human_pose_family.cc
)

set(NMS_LIBS
    ${CMAKE_SOURCE_DIR}/tools/nms/nms.cc
)

set(FACE_GEOMETRY_LIBS
    ${CMAKE_SOURCE_DIR}/tools/face_geometry/face_geometry.cc
    ${CMAKE_SOURCE_DIR}/tools/face_geometry/procrustes_solver.cc
    ${CMAKE_SOURCE_DIR}/tools/face_geometry/geometry_pipeline.cc
    ${CMAKE_SOURCE_DIR}/tools/face_geometry/calculator.cc
)




add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(src)
add_subdirectory(src_unity_api)

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
message(STATUS "    OpenMP_CXX_INCLUDE_DIRS:" "${OpenMP_CXX_INCLUDE_DIRS}"  )
message(STATUS "    ENV{OpenMP_ROOT}: $ENV{OpenMP_ROOT}")

message("")
message(STATUS "  SOURCE: ")
message(STATUS "    SOURCES:" "${SOURCES}" )
message("========================================================================")