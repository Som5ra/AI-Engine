cmake_minimum_required(VERSION 3.4.1)

project(ort_test)


if(BUILD_PLATFORM STREQUAL "linux")
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # set(CMAKE_CXX_STANDARD 17)
    # set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # if(BUILD_PLATFORM STREQUAL "Android")
    #     set(ANDROID_SYSROOT ${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot)
    #     set(OpenCV_DIR "/media/sombrali/HDD1/3d_object_detection/opencv-unity/OpenCV-android-sdk/sdk/native/jni")
    # endif()

    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-static-minimal)
    find_package(OpenCV REQUIRED)

    find_package(OpenMP REQUIRED)
    # message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
    
    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/lib/libonnxruntime.a")
    # set(onnxruntime_DIR "${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/ort-builder/onnxruntime/build_static/Linux/MinSizeRel")
    # find_package(onnxruntime REQUIRED CONFIG PATHS ${onnxruntime_DIR})
    # set(onnxruntime_LIBS "${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/onnxruntime/build_static/Linux/RelWithDebInfo/")


    # set(Eigen3_DIR "/usr/share/eigen3/cmake")
    # find_package(Eigen3 REQUIRED)
    # set(nlohmann_json_DIR "/usr/local/share/cmake/nlohmann_json")
    # find_package(nlohmann_json REQUIRED CONFIG)


    # add_library( ${PROJECT_NAME} SHARED detector2d.h detector2d.cpp  )
    # add_executable( ${PROJECT_NAME} main.cpp )
    # target_link_libraries(  ${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json)


    add_executable( ${PROJECT_NAME} ort_test.cpp )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})

    target_link_libraries( ${PROJECT_NAME}
        ${OpenCV_LIBS} OpenMP::OpenMP_CXX 
        ${onnxruntime_LIBS}
        # ${OpenCV_LIBS} nlohmann_json::nlohmann_json Eigen3::Eigen 
    )
    endif()

if(BUILD_PLATFORM STREQUAL "macos")
    # set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/llvm/19.1.3/bin/clang")
    # set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm/19.1.3/bin/clang++")
    # set(OPENMP_LIBRARIES "/opt/homebrew/Cellar/llvm/19.1.3/lib")
    # set(OPENMP_INCLUDES "/opt/homebrew/Cellar/llvm/19.1.3/include")
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF) #Set this to ON if you want to use GNU++
    set(CMAKE_CXX_STANDARD 17)
    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-build/build/static_lib/libonnxruntime.a")
    add_executable( ${PROJECT_NAME} ort_test.cpp )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
    set(AppleLink "-framework Foundation")
    target_link_libraries( ${PROJECT_NAME}
        ${AppleLink}
        ${onnxruntime_LIBS}
    )

endif()

if(BUILD_PLATFORM STREQUAL "Android")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Android/${ANDROID_ABI}/)
    add_custom_command(TARGET GustoEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:GustoEngine> ${CMAKE_BINARY_DIR}/lib/Android/${ANDROID_ABI}/
    )

    # ----------------------------------------------------------------------------
    #   Install target, for "make install"
    # ----------------------------------------------------------------------------
    install(FILES $<TARGET_FILE:GustoEngine> DESTINATION ${PROJECT_SOURCE_DIR}/../gusto-engine-unity-wrapper/Assets/Plugins/Android/)
endif()

