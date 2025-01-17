cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0072 NEW)
cmake_policy(SET CMP0079 NEW)

# A Sparse Region-Based 3D Object Tracking Approach for the Real World
# =============================================================================
project(pysrt3d LANGUAGES CXX C)

# CXX Standard
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Release")


# Conda Environment
# =============================================================================
if(DEFINED ENV{CONDA_PREFIX})
    message(STATUS "Conda environment detected: $ENV{CONDA_PREFIX}")
    set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    set(CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
endif()


# Python
# =============================================================================
find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)
message(STATUS "Found python ${PYTHON_VERSION_STRING} at: ${PYTHON_EXECUTABLE}")
message(STATUS "Python includes: ${PYTHON_INCLUDE_DIRS}")


# Libraries
# =============================================================================
# find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 3 REQUIRED)
# find_package(OpenCV 4 REQUIRED)
# find_package(OpenMP REQUIRED)
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


# Release settings
# check_cxx_compiler_flag(-Ob3 HAS_OB3)
# check_cxx_compiler_flag(-O3 HAS_O3)
# check_cxx_compiler_flag(-Ob2 HAS_OB2)
# check_cxx_compiler_flag(-O2 HAS_O2)
# if (HAS_OB3)
#     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ob3")
# elseif (HAS_O3)
#     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
# elseif(HAS_OB2)
#     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ob2")
# elseif(HAS_O2)
#     set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
# endif()
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")


# PyBind
# =============================================================================
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11)

# Compiling
# =============================================================================
FILE(GLOB_RECURSE FILES_CPP "${PROJECT_SOURCE_DIR}/src/*.cpp")

## pybind11 module
pybind11_add_module(${PROJECT_NAME}
    ${FILES_CPP}
    "${PROJECT_SOURCE_DIR}/pysrt3d/pysrt3d.cpp"
)

## include
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/pysrt3d
        ${PROJECT_SOURCE_DIR}/third_party
        ${PROJECT_SOURCE_DIR}/third_party/pybind11/include/
        ${PYTHON_INCLUDE_DIRS}
)

## linking
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        ${OpenCV_LIBS}
        OpenGL::GL
        GLEW::GLEW
        glfw
        OpenMP::OpenMP_CXX
        
)

## output
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "pysrt3d" PREFIX "")


# CPP Source Compiling
# =============================================================================
add_library(csrt3d_module SHARED
    ${FILES_CPP}
    "${PROJECT_SOURCE_DIR}/csrt3d/csrt3d.cc"
    "${PROJECT_SOURCE_DIR}/csrt3d/unity_api.cc"
)


## include
target_include_directories(csrt3d_module
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/csrt3d
        ${PROJECT_SOURCE_DIR}/third_party
)

## linking
target_link_libraries(csrt3d_module
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        ${OpenCV_LIBS}
        OpenGL::GL
        GLEW::GLEW
        glfw
        OpenMP::OpenMP_CXX
)

# example
# =============================================================================

## example
add_executable(run_on_webcam
    ${FILES_CPP}
    "${PROJECT_SOURCE_DIR}/csrt3d/csrt3d.cc"
    "${PROJECT_SOURCE_DIR}/csrt3d/example/run_on_webcam.cc"
)

## include
target_include_directories(run_on_webcam
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/csrt3d
        ${PROJECT_SOURCE_DIR}/third_party
)



## linking
target_link_libraries(run_on_webcam
    PUBLIC
        # csrt3d_module
        Eigen3::Eigen
    PRIVATE
        ${OpenCV_LIBS}
        OpenGL::GL
        GLEW::GLEW
        glfw
        OpenMP::OpenMP_CXX
)



# Unity Plugin Compiling
# =============================================================================


# set(FILES_CPP_WO_OPENGL src/body.cpp src/camera.cpp src/common.cpp src/model.cpp src/region_modality.cpp src/tracker.cpp)


add_library(model_target SHARED
    # ${FILES_CPP_WO_OPENGL}
    ${FILES_CPP}
    "${PROJECT_SOURCE_DIR}/csrt3d/csrt3d.cc"
    "${PROJECT_SOURCE_DIR}/csrt3d/unity_api.cc"
)

# target_compile_definitions(model_target PRIVATE __DISABLE_OPENGL__)


## include
target_include_directories(model_target
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/csrt3d
        ${PROJECT_SOURCE_DIR}/third_party
)

## linking
target_link_libraries(model_target
    PUBLIC
        Eigen3::Eigen
    PRIVATE
        ${OpenCV_LIBS}
        OpenGL::GL
        GLEW::GLEW
        glfw
        OpenMP::OpenMP_CXX
)

install(FILES $<TARGET_FILE:model_target> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Linux)



