cmake_minimum_required(VERSION 3.20.0)

project(detection_example)

add_definitions("-DBUILD_PLATFORM_${BUILD_PLATFORM}")
add_definitions("-DDEBUG")





if(BUILD_PLATFORM STREQUAL "linux")
    include_directories(${CMAKE_SOURCE_DIR}/include )
    add_executable( ${PROJECT_NAME} detection_example.cc ${SOURCES} ${DETECTION_2D_LIBS} ${NMS_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})

    # add_library( detection_csharp_example SHARED detection_example_mobile.cc ${SOURCES} ${DETECTION_2D_LIBS} ${NMS_LIBS})
    # target_include_directories(detection_csharp_example PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    # target_link_libraries( detection_csharp_example OpenMP::OpenMP_CXX ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})

    # add_executable( headband_tracking_example headband_tracking_example.cc ${SOURCES} ${DETECTION_2D_LIBS} ${NMS_LIBS} ${FACE_MODEL_LIBS} ${FACE_GEOMETRY_LIBS})
    # target_include_directories(headband_tracking_example PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
    # target_link_libraries( headband_tracking_example OpenMP::OpenMP_CXX ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()

if (BUILD_PLATFORM STREQUAL "macos")
    include_directories(${CMAKE_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
    # find_package(nlohmann_json REQUIRED)
    add_executable( ${PROJECT_NAME} detection_example.cc ${SOURCES} ${DETECTION_2D_LIBS} ${NMS_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX ${AppleLink} ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()

# if (BUILD_PLATFORM STREQUAL "android")
#     include_directories(${CMAKE_SOURCE_DIR}/include)
#     add_library( detection_csharp_example SHARED detection_example_mobile.cc ${SOURCES} ${DETECTION_2D_LIBS} ${NMS_LIBS})
#     target_include_directories(detection_csharp_example PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
#     target_link_libraries( detection_csharp_example OpenMP::OpenMP_CXX ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})
# endif()

# if(BUILD_PLATFORM STREQUAL "ios")
#     include_directories(${CMAKE_SOURCE_DIR}/include )
#     # add_executable( ${PROJECT_NAME} face_geometry_example.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp ${CMAKE_SOURCE_DIR}/tools/nms/nms.cpp ${CMAKE_SOURCE_DIR}/src/BaseONNX.cpp ${CMAKE_SOURCE_DIR}/src/face_model_family.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/face_geometry.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/procrustes_solver.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/geometry_pipeline.cpp ${CMAKE_SOURCE_DIR}/tools/face_geometry/calculator.cpp)
    
#     add_library( detection_csharp_example STATIC detection_example_mobile.cc  ${SOURCES} ${DETECTION_2D_LIBS} ${NMS_LIBS})
#     target_include_directories(detection_csharp_example PRIVATE ${OPENMP_INCLUDES} ${onnxruntime_INCLUDE_DIRS})
#     target_link_libraries( detection_csharp_example "-framework Foundation" ${onnxruntime_LIBS} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})


#     # make a bundle
#     # cannot auto-configure ${EFFECTIVE_PLATFORM_NAME} in $<TARGET_FILE:xxx>, so hardcode it here
#     # pls see https://cmake.org/pipermail/cmake/2016-March/063050.html
#     set(STATUS custom_target_path ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-"iphoneos"/detection_csharp_example.a)
#     add_custom_command(TARGET detection_csharp_example POST_BUILD
#         COMMAND ${libtool} -static -o libfat_example.a ${custom_target_path} ${OpenCV_LIB_FILES} ${onnxruntime_LIBS}
#         COMMENT "Bundling detection_csharp_example"
#         VERBATIM)
# endif()


# [Sombra] -> need msvc compiler
# if(BUILD_PLATFORM STREQUAL "windows")
#     include_directories(${CMAKE_SOURCE_DIR}/include)
#     add_executable( ${PROJECT_NAME} detection_example.cpp ${SOURCES} ${DETECTION_2D_LIBS})
#     target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
#     target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${onnxruntime_LIBS}  ${GLOBAL_LINK_3RD_PARTY_LIBS})
# endif()


# if(BUILD_PLATFORM STREQUAL "linux")
#     # file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Linux/)
#     add_custom_command(TARGET detection_csharp_example  POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:detection_csharp_example> ${CMAKE_BINARY_DIR}/lib/Linux/
#     )
#     install(FILES $<TARGET_FILE:detection_csharp_example> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Linux)
# endif()