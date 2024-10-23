cmake_minimum_required(VERSION 3.4.1)

project(GustoEngineExamples)


if(BUILD_PLATFORM STREQUAL "linux")


    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    find_package(Eigen3 REQUIRED)
    find_package(glog REQUIRED)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-static-minimal)

    find_package(OpenCV REQUIRED CONFIG PATHS ${OpenCV_DIR})
    # include_directories(${OpenCV_INCLUDE_DIRS})
    
    find_package(OpenMP REQUIRED)

    set(nlohmann_json_DIR "/usr/local/share/cmake/nlohmann_json")
    find_package(nlohmann_json REQUIRED CONFIG)

    include_directories(${CMAKE_SOURCE_DIR}/include)

    add_executable( ${PROJECT_NAME} yolov7_tiny_benchmark.cpp ${SOURCES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)
endif()
