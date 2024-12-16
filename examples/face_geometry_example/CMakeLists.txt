cmake_minimum_required(VERSION 3.20.0)

project(face_geometry_example)

add_definitions("-DBUILD_PLATFORM_${BUILD_PLATFORM}")



if(BUILD_PLATFORM STREQUAL "linux")
    add_definitions("-DDEBUG")
    include_directories(${CMAKE_SOURCE_DIR}/include )
    # add_executable( ${PROJECT_NAME} face_geometry_example.cc ${CMAKE_SOURCE_DIR}/src/utils.cpp ${CMAKE_SOURCE_DIR}/tools/nms/nms.cpp ${CMAKE_SOURCE_DIR}/src/BaseONNX.cpp ${CMAKE_SOURCE_DIR}/src/face_model_family.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/face_geometry.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/procrustes_solver.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/geometry_pipeline.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/calculator.cpp)
    add_executable( ${PROJECT_NAME} face_geometry_example.cc ${SOURCES} ${FACE_MODEL_LIBS} ${NMS_LIBS} ${FACE_GEOMETRY_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()

if (BUILD_PLATFORM STREQUAL "macos")

    include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
    add_executable( ${PROJECT_NAME} face_geometry_example.cc ${SOURCES} ${FACE_MODEL_LIBS} ${NMS_LIBS} ${FACE_GEOMETRY_LIBS})

    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX ${AppleLink} ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})

endif()


# [Sombra] -> need msvc compiler
if(BUILD_PLATFORM STREQUAL "windows")
    include_directories(${CMAKE_SOURCE_DIR}/include)
    add_executable( ${PROJECT_NAME} face_geometry_example.cc ${SOURCES} ${FACE_MODEL_LIBS} ${NMS_LIBS} ${FACE_GEOMETRY_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${onnxruntime_LIBS}  ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()