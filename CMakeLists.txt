cmake_minimum_required(VERSION 3.4.1)

project(GustoEngine)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ANDROID_SYSROOT ${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot)



set(OpenCV_DIR "/media/sombrali/HDD1/3d_object_detection/opencv-unity/OpenCV-android-sdk/sdk/native/jni")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")


set(nlohmann_json_DIR "/usr/local/share/cmake/nlohmann_json")
find_package(nlohmann_json REQUIRED CONFIG)



add_library( ${PROJECT_NAME} SHARED detector2d.h detector2d.cpp  )


target_link_libraries( GustoEngine ${OpenCV_LIBS} nlohmann_json::nlohmann_json)



set_target_properties(GustoEngine PROPERTIES PREFIX "lib")
set_target_properties(GustoEngine PROPERTIES SUFFIX ".so")

# ----------------------------------------------------------------------------
#   Install target, for "make install"
# ----------------------------------------------------------------------------
install(FILES $<TARGET_FILE:GustoEngine> DESTINATION ${PROJECT_SOURCE_DIR}/../gusto-engine-unity-wrapper/Assets/Plugins/Android/)