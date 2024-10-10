cmake_minimum_required(VERSION 3.4.1)

project(GustoEngine)


# set(BUILD_PLATFORM "android")
message(STATUS "BUILD_PLATFORM: ${BUILD_PLATFORM}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_EXPORT_COMPILE_COMMANDS: ${CMAKE_EXPORT_COMPILE_COMMANDS}")


# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_CXX_STANDARD 17)

if(BUILD_PLATFORM STREQUAL "android")
    set(ANDROID_SYSROOT ${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv_source/opencv-4.10.0-android-sdk/OpenCV-android-sdk/sdk/native/jni/abi-arm64-v8a)
elseif(BUILD_PLATFORM STREQUAL "linux")
    find_package(Eigen3 REQUIRED)
    find_package(glog REQUIRED)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv_source/opencv-static-build)
endif()

find_package(OpenCV REQUIRED CONFIG PATHS ${OpenCV_DIR})
# include_directories(${OpenCV_INCLUDE_DIRS})

set(nlohmann_json_DIR "/usr/local/share/cmake/nlohmann_json")
find_package(nlohmann_json REQUIRED CONFIG)


add_library( ${PROJECT_NAME} SHARED detector2d.h detector2d.cpp  )


target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json)


# file(COPY $<TARGET_FILE:GustoEngine>
#      DESTINATION ${CMAKE_BINARY_DIR}/build/Android/${ANDROID_ABI})
# get_filename_component(targetfile "$<TARGET_FILE:GustoEngine>" NAME)

# message(${CMAKE_BINARY_DIR}/Android/${ANDROID_ABI}/${targetfile})

if(BUILD_PLATFORM STREQUAL "android")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Android/${ANDROID_ABI}/)
    add_custom_command(TARGET GustoEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:GustoEngine> ${CMAKE_BINARY_DIR}/lib/Android/${ANDROID_ABI}/
    )

    # ----------------------------------------------------------------------------
    #   Install target, for "make install"
    # ----------------------------------------------------------------------------
    # install(FILES $<TARGET_FILE:GustoEngine> DESTINATION ${PROJECT_SOURCE_DIR}/../gusto-engine-unity-wrapper/Assets/Plugins/Android/)
    install(FILES $<TARGET_FILE:GustoEngine> DESTINATION /media/sombrali/HDD1/3d_object_detection/opencv-unity/gusto-engine-unity-wrapper1/Assets/Plugins/Android)
elseif(BUILD_PLATFORM STREQUAL "linux")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Linux/)
    add_custom_command(TARGET GustoEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:GustoEngine> ${CMAKE_BINARY_DIR}/lib/Linux/
    )
    # install(FILES $<TARGET_FILE:GustoEngine> DESTINATION ${PROJECT_SOURCE_DIR}/../gusto-engine-unity-wrapper/Assets/Plugins/Linux/)
    install(FILES $<TARGET_FILE:GustoEngine> DESTINATION /media/sombrali/HDD1/3d_object_detection/opencv-unity/gusto-engine-unity-wrapper1/Assets/Plugins/Linux)
endif()


message("")
message("Compile Summary:")
message("========================================================================")
# ============================ Options ===========================
message("")

message(STATUS "  Build options: ")
message(STATUS "    BUILD_PLATFORM:"           ${BUILD_PLATFORM}      )

# ============================ OpenCV ===========================
message("")
message(STATUS "  OpenCV: ")
message(STATUS "    Version:" "${OpenCV_VERSION}" )
message(STATUS "    OpenCV_DIR:" "${OpenCV_DIR}"  )
# message(STATUS "    OpenCV_LIBS:" "${OpenCV_LIBS}" )
# message(STATUS "    OpenCV_INCLUDE_DIRS:" "${OpenCV_INCLUDE_DIRS}" )

message("")
message("========================================================================")