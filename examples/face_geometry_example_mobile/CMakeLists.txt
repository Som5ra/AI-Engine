cmake_minimum_required(VERSION 3.20.0)

project(face_geometry_example_mobile)

add_definitions("-DBUILD_PLATFORM_${BUILD_PLATFORM}")

file(GLOB FACE_GEOMETRY_LIBS ${CMAKE_SOURCE_DIR}/tools/face_geometry/*.cpp)
file(GLOB NMS_LIBS ${CMAKE_SOURCE_DIR}/tools/nms/*.cpp)

if(BUILD_PLATFORM STREQUAL "ios")
    include_directories(${CMAKE_SOURCE_DIR}/include )
    # add_executable( ${PROJECT_NAME} face_geometry_example.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp ${CMAKE_SOURCE_DIR}/tools/nms/nms.cpp ${CMAKE_SOURCE_DIR}/src/BaseONNX.cpp ${CMAKE_SOURCE_DIR}/src/face_model_family.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/face_geometry.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/procrustes_solver.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/geometry_pipeline.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/calculator.cpp)
    add_library( ${PROJECT_NAME} STATIC face_geometry_example_mobile.cpp ${SOURCES} ${NMS_LIBS} ${FACE_GEOMETRY_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} "-framework Foundation" ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})


    # make a bundle
    # cannot auto-configure ${EFFECTIVE_PLATFORM_NAME} in $<TARGET_FILE:xxx>, so hardcode it here
    # pls see https://cmake.org/pipermail/cmake/2016-March/063050.html
    set(STATUS custom_target_path ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-"iphoneos"/${PROJECT_NAME}.a)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${libtool} -static -o libfat_example.a ${custom_target_path} ${OpenCV_LIB_FILES} ${onnxruntime_LIBS}
        COMMENT "Bundling ${PROJECT_NAME}"
        VERBATIM)
endif()

if(BUILD_PLATFORM STREQUAL "linux")
    include_directories(${CMAKE_SOURCE_DIR}/include )
    # add_executable( ${PROJECT_NAME} face_geometry_example.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp ${CMAKE_SOURCE_DIR}/tools/nms/nms.cpp ${CMAKE_SOURCE_DIR}/src/BaseONNX.cpp ${CMAKE_SOURCE_DIR}/src/face_model_family.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/face_geometry.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/procrustes_solver.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/geometry_pipeline.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/calculator.cpp)
    add_library( ${PROJECT_NAME} SHARED face_geometry_example_mobile.cpp ${SOURCES} ${CMAKE_SOURCE_DIR}/tools/nms/nms.cpp ${FACE_GEOMETRY_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS} OpenMP::OpenMP_CXX)

endif()
