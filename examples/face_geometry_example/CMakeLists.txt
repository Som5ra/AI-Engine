cmake_minimum_required(VERSION 3.20.0)

project(face_geometry_example)


if(BUILD_PLATFORM STREQUAL "linux")
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


    find_package (Eigen3 REQUIRED)    
    find_package(OpenMP REQUIRED)


    # set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-mobile-4.10.0-ubuntu-2204/lib/cmake/opencv4)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-static-full)
    # set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-static-minimal)
    find_package(OpenCV REQUIRED)
    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-linux-x64-static_lib-1.19.2/lib/libonnxruntime.a")

    include_directories(${CMAKE_SOURCE_DIR}/include)
    message(STATUS "Include: ${CMAKE_SOURCE_DIR}/include")
    # find_package(nlohmann_json REQUIRED)
    add_executable( ${PROJECT_NAME} face_geometry_example.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp ${CMAKE_SOURCE_DIR}/tools/nms/nms.cpp ${CMAKE_SOURCE_DIR}/src/BaseONNX.cpp ${CMAKE_SOURCE_DIR}/src/face_model_family.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/face_geometry.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/procrustes_solver.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/geometry_pipeline.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/calculator.cpp)
    
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX Eigen3::Eigen ${onnxruntime_LIBS} ${OpenCV_LIBS})


endif()

if (BUILD_PLATFORM STREQUAL "macos")
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


    find_package(Eigen3 REQUIRED)    
    find_package(OpenMP REQUIRED)


    # set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-mobile-4.10.0-ubuntu-2204/lib/cmake/opencv4)
    set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-minimal-macos)
    # set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/opencv-static-minimal)
    find_package(OpenCV REQUIRED)
    set(onnxruntime_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2/include")
    set(onnxruntime_LIBS "${CMAKE_SOURCE_DIR}/3rdparty/onnxruntime/onnxruntime-osx-arm64-static_lib-1.19.2/lib/libonnxruntime.a")

    include_directories(${CMAKE_SOURCE_DIR}/include)
    message(STATUS "Include: ${CMAKE_SOURCE_DIR}/include")
    # find_package(nlohmann_json REQUIRED)
    add_executable( ${PROJECT_NAME} face_geometry_example.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp ${CMAKE_SOURCE_DIR}/tools/nms/nms.cpp ${CMAKE_SOURCE_DIR}/src/BaseONNX.cpp ${CMAKE_SOURCE_DIR}/src/face_model_family.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/face_geometry.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/procrustes_solver.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/geometry_pipeline.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/calculator.cpp)

    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX Eigen3::Eigen ${onnxruntime_LIBS} ${OpenCV_LIBS})


endif()