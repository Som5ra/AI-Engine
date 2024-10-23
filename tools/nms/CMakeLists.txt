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
    set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/llvm/19.1.2/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm/19.1.2/bin/clang++")
    set(OPENMP_LIBRARIES "/opt/homebrew/Cellar/llvm/19.1.2/lib")
    set(OPENMP_INCLUDES "/opt/homebrew/Cellar/llvm/19.1.2/include")
    
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY omp)
    endif()
endif()

find_package(OpenMP REQUIRED)
# enable omp.h
if(BUILD_PLATFORM STREQUAL "macos")
    # set(OpenMP_OMPH "/opt/homebrew/opt/libomp/include")
    include_directories(${OPENMP_INCLUDES})
    message(STATUS ${OPENMP_INCLUDES})
endif()

# add_executable( ${PROJECT_NAME} nms.cpp)
add_library( ${PROJECT_NAME} SHARED nms.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES})
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
elseif(BUILD_PLATFORM STREQUAL "macos")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/MacOS/)
    add_custom_command(TARGET nms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nms> ${CMAKE_BINARY_DIR}/lib/MacOS/
    )
    install(FILES $<TARGET_FILE:nms> DESTINATION /media/sombrali/HDD1/3d_object_detection/opencv-unity/gusto-engine-unity-wrapper1/Assets/Plugins/MacOS)
endif()