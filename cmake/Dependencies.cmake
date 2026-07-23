set(AI_ENGINE_3RDPARTY_DIR "${CMAKE_SOURCE_DIR}/3rdparty" CACHE PATH
    "Directory containing prebuilt third-party dependencies")

if(BUILD_PLATFORM STREQUAL "linux")
    set(_opencv_default
        "${AI_ENGINE_3RDPARTY_DIR}/opencv/opencv-mobile-4.10.0-ubuntu-2204/lib/cmake/opencv4")
    set(_ort_include_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/include")
    set(_ort_library_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/lib/libonnxruntime.a")
elseif(BUILD_PLATFORM STREQUAL "android")
    if(NOT ANDROID_ABI)
        message(FATAL_ERROR "ANDROID_ABI is required for an Android build")
    endif()
    set(AI_ENGINE_ANDROID_ABIS armeabi-v7a arm64-v8a x86_64)
    list(FIND AI_ENGINE_ANDROID_ABIS "${ANDROID_ABI}" AI_ENGINE_ANDROID_ABI_INDEX)
    if(AI_ENGINE_ANDROID_ABI_INDEX EQUAL -1)
        message(FATAL_ERROR
            "Unsupported Android ABI='${ANDROID_ABI}'. "
            "Choose one of: ${AI_ENGINE_ANDROID_ABIS}")
    endif()
    set(_opencv_default
        "${AI_ENGINE_3RDPARTY_DIR}/opencv/opencv-mobile-4.10.0-android/sdk/native/jni/abi-${ANDROID_ABI}")
    set(_ort_include_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime-android-static_lib-1.19.2/static_lib_${ANDROID_ABI}/include")
    set(_ort_library_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime-android-static_lib-1.19.2/static_lib_${ANDROID_ABI}/lib/libonnxruntime.a")
elseif(BUILD_PLATFORM STREQUAL "macos")
    set(_opencv_default
        "${AI_ENGINE_3RDPARTY_DIR}/opencv/opencv-mobile-4.10.0-macos/lib/cmake/opencv4")
    set(_ort_include_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2/include")
    set(_ort_library_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2/lib/libonnxruntime.a")
elseif(BUILD_PLATFORM STREQUAL "ios")
    set(_opencv_default
        "${AI_ENGINE_3RDPARTY_DIR}/opencv/opencv-mobile-4.10.0-ios/lib/cmake/opencv4")
    set(_ort_include_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime.xcframework-1.19.2/onnxruntime.xcframework/Headers")
    set(_ort_library_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime.xcframework-1.19.2/onnxruntime.xcframework/ios-arm64/libonnxruntime.a")
elseif(BUILD_PLATFORM STREQUAL "windows")
    set(_opencv_default
        "${AI_ENGINE_3RDPARTY_DIR}/opencv/opencv-mobile-4.10.0/build/install/x64/vc17/staticlib")
    set(_ort_include_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime-win-x64-static_lib-1.19.2/include")
    set(_ort_library_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime-win-x64-static_lib-1.19.2/lib/onnxruntime.lib")
elseif(BUILD_PLATFORM STREQUAL "wasm")
    set(_opencv_default
        "${AI_ENGINE_3RDPARTY_DIR}/opencv/opencv-mobile-4.10.0-webassembly/simd/lib/cmake/opencv4")
    set(_ort_include_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime-wasm-static_lib-simd-1.17.1/include")
    set(_ort_library_default
        "${AI_ENGINE_3RDPARTY_DIR}/onnxruntime/onnxruntime-wasm-static_lib-simd-1.17.1/lib/libonnxruntime.a")
endif()

set(OpenCV_DIR "${_opencv_default}" CACHE PATH "Directory containing OpenCVConfig.cmake")
set(onnxruntime_INCLUDE_DIRS "${_ort_include_default}" CACHE PATH
    "Directory containing onnxruntime_cxx_api.h")
set(onnxruntime_LIBS "${_ort_library_default}" CACHE FILEPATH
    "Path to the ONNX Runtime library")

if(NOT EXISTS "${OpenCV_DIR}")
    message(FATAL_ERROR
        "OpenCV was not found at '${OpenCV_DIR}'. "
        "Bootstrap the pinned dependencies or set OpenCV_DIR explicitly.")
endif()

if(NOT EXISTS "${onnxruntime_INCLUDE_DIRS}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime headers were not found at '${onnxruntime_INCLUDE_DIRS}'. "
        "Bootstrap the pinned dependencies or set onnxruntime_INCLUDE_DIRS explicitly.")
endif()

if(NOT EXISTS "${onnxruntime_LIBS}")
    message(FATAL_ERROR
        "ONNX Runtime library was not found at '${onnxruntime_LIBS}'. "
        "Bootstrap the pinned dependencies or set onnxruntime_LIBS explicitly.")
endif()

find_package(OpenCV REQUIRED CONFIG)

add_library(ai_engine_onnxruntime UNKNOWN IMPORTED GLOBAL)
set_target_properties(ai_engine_onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_LIBS}"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIRS}"
)

set(AI_ENGINE_OPENMP_TARGET "")
if(NOT BUILD_PLATFORM STREQUAL "ios"
   AND NOT BUILD_PLATFORM STREQUAL "windows"
   AND NOT BUILD_PLATFORM STREQUAL "wasm")
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        set(AI_ENGINE_OPENMP_TARGET OpenMP::OpenMP_CXX)
        add_compile_definitions(AI_ENGINE_HAS_OPENMP=1)
    else()
        message(WARNING "OpenMP was not found; inference loops will run serially")
    endif()
endif()

set(GLOBAL_LINK_3RD_PARTY_LIBS
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)

set(AI_ENGINE_LINK_LIBRARIES
    ai_engine_onnxruntime
    ${OpenCV_LIBS}
    ${GLOBAL_LINK_3RD_PARTY_LIBS}
    ${AI_ENGINE_OPENMP_TARGET}
)
