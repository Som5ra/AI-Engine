cmake_minimum_required(VERSION 3.20.0)

project(face_geometry_example_mobile)

add_definitions("-DBUILD_PLATFORM_${BUILD_PLATFORM}")


file(GLOB FACE_GEOMETRY_LIBS ${CMAKE_SOURCE_DIR}/tools/face_geometry/*.cpp)

if(BUILD_PLATFORM STREQUAL "ios")
    include_directories(${CMAKE_SOURCE_DIR}/include )
    # add_executable( ${PROJECT_NAME} face_geometry_example.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp ${CMAKE_SOURCE_DIR}/tools/nms/nms.cpp ${CMAKE_SOURCE_DIR}/src/BaseONNX.cpp ${CMAKE_SOURCE_DIR}/src/face_model_family.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/face_geometry.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/procrustes_solver.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/geometry_pipeline.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/calculator.cpp)
    add_executable( ${PROJECT_NAME} face_geometry_example.cpp ${SOURCES} ${CMAKE_SOURCE_DIR}/tools/nms/nms.cpp ${FACE_GEOMETRY_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()
