cmake_minimum_required(VERSION 3.4.1)

project(GustoEngineExamples)


if(BUILD_PLATFORM STREQUAL "linux")

    find_package(glog REQUIRED)


    include_directories(${CMAKE_SOURCE_DIR}/include)

    add_executable( ${PROJECT_NAME} yolov7_tiny_benchmark.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)
endif()
