cmake_minimum_required(VERSION 3.10)
add_definitions("-DBUILD_PLATFORM_${def}")
# CXX Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Release")


# Libraries
# =============================================================================
if(BUILD_PLATFORM STREQUAL "android" )
    add_compile_definitions(__DISABLE_OPENGL__)
    message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
    set(FILES_CPP ${CMAKE_CURRENT_SOURCE_DIR}/src/body.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/camera.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/common.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/model.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/region_modality.cpp 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/tracker.cpp
    )
else()

    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(glfw3 3 REQUIRED)

    add_compile_definitions(__OPENCV_DEBUG__)

    FILE(GLOB_RECURSE FILES_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
    set(Custom_Linker OpenGL::GL GLEW::GLEW glfw)

endif()
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(OpenMP REQUIRED)
# if(OpenMP_FOUND)
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# endif()


# Compiler Settings
# =============================================================================
# Default settings
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march HAS_MARCH)
check_cxx_compiler_flag(-mtune HAS_MTUNE)
if(HAS_MARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
if(HAS_MTUNE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native")
endif()


# Compiling


# CPP Source Compiling
# =============================================================================
add_library(csrt3d_module SHARED
    ${FILES_CPP}
    "${CMAKE_CURRENT_SOURCE_DIR}/csrt3d/csrt3d.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrt3d/unity_api.cc"
)


## include
target_include_directories(csrt3d_module
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/csrt3d
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

## linking
target_link_libraries(csrt3d_module
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        ${OpenCV_LIBS}
        # OpenGL::GL
        # GLEW::GLEW
        # glfw
        ${Custom_Linker}
        OpenMP::OpenMP_CXX

    )

# example
# =============================================================================

if(NOT BUILD_PLATFORM STREQUAL "android" )


    add_executable(run_on_webcam
        ${FILES_CPP}
        "${CMAKE_CURRENT_SOURCE_DIR}/csrt3d/csrt3d.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrt3d/example/run_on_webcam.cc"
    )

    ## include
    target_include_directories(run_on_webcam
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/csrt3d
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )



    ## linking
    target_link_libraries(run_on_webcam
        PUBLIC
            # csrt3d_module
            Eigen3::Eigen
        PRIVATE
            ${OpenCV_LIBS}
            # OpenGL::GL
            # GLEW::GLEW
            # glfw
            ${Custom_Linker}
            OpenMP::OpenMP_CXX
    )

endif()

# Unity Plugin Compiling
# =============================================================================


# set(FILES_CPP_WO_OPENGL src/body.cpp src/camera.cpp src/common.cpp src/model.cpp src/region_modality.cpp src/tracker.cpp)


add_library(model_target SHARED
    # ${FILES_CPP_WO_OPENGL}
    ${FILES_CPP}
    "${CMAKE_CURRENT_SOURCE_DIR}/csrt3d/csrt3d.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrt3d/unity_api.cc"
)

# target_compile_definitions(model_target PRIVATE __DISABLE_OPENGL__)


## include
target_include_directories(model_target
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/csrt3d
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

## linking
target_link_libraries(model_target
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        ${OpenCV_LIBS}
        # OpenGL::GL
        # GLEW::GLEW
        # glfw
        ${Custom_Linker}
        OpenMP::OpenMP_CXX
)


if(BUILD_PLATFORM STREQUAL "linux" )
    install(FILES $<TARGET_FILE:model_target> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Linux)
elseif(BUILD_PLATFORM STREQUAL "android" )
    install(FILES $<TARGET_FILE:model_target> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Android/${ANDROID_ABI})
endif()