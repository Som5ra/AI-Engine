cmake_minimum_required(VERSION 3.20.0)

project(nms)

if(BUILD_PLATFORM STREQUAL "android")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/aarch64/libomp.a")
endif()

if(BUILD_PLATFORM STREQUAL "linux")
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(BUILD_PLATFORM STREQUAL "macos")
endif()

find_package(OpenMP REQUIRED)
# add_executable( ${PROJECT_NAME} nms.cpp)
add_library( ${PROJECT_NAME} SHARED nms.cpp)
target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX)




# ----------------------------------------------------------------------------
#   Install target, for "make install"
# ----------------------------------------------------------------------------
if(BUILD_PLATFORM STREQUAL "android")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Android/${ANDROID_ABI}/)
    add_custom_command(TARGET nms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nms> ${CMAKE_BINARY_DIR}/lib/Android/${ANDROID_ABI}/
    )
    install(FILES $<TARGET_FILE:nms> DESTINATION /media/sombrali/HDD1/3d_object_detection/opencv-unity/gusto-engine-unity-wrapper1/Assets/Plugins/Android)
elseif(BUILD_PLATFORM STREQUAL "linux")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Linux/)
    add_custom_command(TARGET nms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nms> ${CMAKE_BINARY_DIR}/lib/Linux/
    )
    install(FILES $<TARGET_FILE:nms> DESTINATION /media/sombrali/HDD1/3d_object_detection/opencv-unity/gusto-engine-unity-wrapper1/Assets/Plugins/Linux)
endif()