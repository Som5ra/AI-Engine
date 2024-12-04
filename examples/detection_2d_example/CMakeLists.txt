cmake_minimum_required(VERSION 3.20.0)

project(detection_example)

add_definitions("-DBUILD_PLATFORM_${BUILD_PLATFORM}")
add_definitions("-DDEBUG")





if(BUILD_PLATFORM STREQUAL "linux")
    include_directories(${CMAKE_SOURCE_DIR}/include )
    add_executable( ${PROJECT_NAME} detection_example.cpp ${SOURCES} ${DETECTION_2D_LIBS} ${NMS_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()

if (BUILD_PLATFORM STREQUAL "macos")
    include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
    # find_package(nlohmann_json REQUIRED)
    add_executable( ${PROJECT_NAME} detection_example.cpp ${SOURCES} ${DETECTION_2D_LIBS} ${NMS_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX ${AppleLink} ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()

if (BUILD_PLATFORM STREQUAL "android")
    include_directories(${CMAKE_SOURCE_DIR}/include)
    add_library( detection_csharp_example SHARED detection_example_mobile.cc ${SOURCES} ${DETECTION_2D_LIBS} ${NMS_LIBS})
    target_include_directories(detection_csharp_example PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( detection_csharp_example OpenMP::OpenMP_CXX ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()

# [Sombra] -> need msvc compiler
# if(BUILD_PLATFORM STREQUAL "windows")
#     include_directories(${CMAKE_SOURCE_DIR}/include)
#     add_executable( ${PROJECT_NAME} detection_example.cpp ${SOURCES} ${DETECTION_2D_LIBS})
#     target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
#     target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${onnxruntime_LIBS}  ${GLOBAL_LINK_3RD_PARTY_LIBS})
# endif()