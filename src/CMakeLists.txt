cmake_minimum_required(VERSION 3.20.0)

project(GustoEngine)

if(BUILD_PLATFORM STREQUAL "android")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g0")
    set(ANDROID_SYSROOT ${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot)
    message(STATUS "abi: ${ANDROID_ABI}")

    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")

    if (ANDROID_ABI STREQUAL "arm64-v8a")
        set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-4.10.0-android-sdk/OpenCV-android-sdk/sdk/native/jni/abi-arm64-v8a)
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/aarch64/libomp.a")

    elseif (ANDROID_ABI STREQUAL "armeabi-v7a")
        set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-4.10.0-android-sdk/OpenCV-android-sdk/sdk/native/jni/abi-armeabi-v7a)
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/arm/libomp.a")

    elseif(ANDROID_ABI STREQUAL "x86")
        set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-4.10.0-android-sdk/OpenCV-android-sdk/sdk/native/jni/abi-x86)
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/i386/libomp.a")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-4.10.0-android-sdk/OpenCV-android-sdk/sdk/native/jni/abi-x86_64)
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/x86_64/libomp.a")
    endif()

    find_package(OpenMP REQUIRED)
    include_directories("/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/include")
elseif(BUILD_PLATFORM STREQUAL "linux")
    
    find_package(Eigen3 REQUIRED)
    find_package(glog REQUIRED)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-static-minimal)
    # set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-mobile-4.10.0-ubuntu-2204/lib/cmake/opencv4)

    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/lib/libonnxruntime.a")
    message(STATUS "onnxruntime_INCLUDE_DIRS: ${onnxruntime_INCLUDE_DIRS}")
    message(STATUS "onnxruntime_LIBS: ${onnxruntime_LIBS}")
    find_package(OpenMP REQUIRED)
elseif(BUILD_PLATFORM STREQUAL "macos")
    set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/llvm/19.1.3/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm/19.1.3/bin/clang++")
    set(OPENMP_LIBRARIES "/opt/homebrew/Cellar/llvm/19.1.3/lib")
    set(OPENMP_INCLUDES "/opt/homebrew/Cellar/llvm/19.1.3/include")
    
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
    endif()
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.a")
    find_package(OpenMP REQUIRED)
    include_directories(${OPENMP_INCLUDES})
    message(STATUS ${OPENMP_INCLUDES})
endif()

find_package(OpenCV REQUIRED CONFIG PATHS ${OpenCV_DIR})
# include_directories(${OpenCV_INCLUDE_DIRS})

set(nlohmann_json_DIR "/usr/local/share/cmake/nlohmann_json")
find_package(nlohmann_json REQUIRED CONFIG)

include_directories(${CMAKE_SOURCE_DIR}/include)


add_library( ${PROJECT_NAME} SHARED detector2d.cpp ${SOURCES})

if(BUILD_PLATFORM STREQUAL "linux")
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
    # target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX ${onnxruntime_LIBS})
    # opencv protobuf conflicts with onnxruntime protobuf
    # while DNN/ONNX: Build OpenCV with Protobuf to import ONNX models in function 'readNetFromONNX'
    # so there can be only one module built in our engine between OPENCV::DNN and ONNX
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX )
elseif(BUILD_PLATFORM STREQUAL "android")
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json ${OpenMP_omp_LIBRARY})
elseif(BUILD_PLATFORM STREQUAL "macos")
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)
endif()
# file(COPY $<TARGET_FILE:GustoEngine>
#      DESTINATION ${CMAKE_BINARY_DIR}/build/Android/${ANDROID_ABI})
# get_filename_component(targetfile "$<TARGET_FILE:GustoEngine>" NAME)

# message(${CMAKE_BINARY_DIR}/Android/${ANDROID_ABI}/${targetfile})

if(BUILD_PLATFORM STREQUAL "android")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/)
    add_custom_command(TARGET GustoEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:GustoEngine> ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/
    )

    # ----------------------------------------------------------------------------
    #   Install target, for "make install"
    # ----------------------------------------------------------------------------
    # install(FILES $<TARGET_FILE:GustoEngine> DESTINATION ${PROJECT_SOURCE_DIR}/../gusto-engine-unity-wrapper/Assets/Plugins/Android/)
    install(FILES $<TARGET_FILE:GustoEngine> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Android/${ANDROID_ABI})
elseif(BUILD_PLATFORM STREQUAL "linux")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Linux/)
    add_custom_command(TARGET GustoEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:GustoEngine> ${CMAKE_BINARY_DIR}/lib/Linux/
    )
    # install(FILES $<TARGET_FILE:GustoEngine> DESTINATION ${PROJECT_SOURCE_DIR}/../gusto-engine-unity-wrapper/Assets/Plugins/Linux/)
    install(FILES $<TARGET_FILE:GustoEngine> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Linux)
endif()


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