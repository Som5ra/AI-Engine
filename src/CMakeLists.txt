cmake_minimum_required(VERSION 3.20.0)

project(GustoEngine)





include_directories(${CMAKE_SOURCE_DIR}/include)

if(BUILD_PLATFORM STREQUAL "android")
    include_directories(${onnxruntime_INCLUDE_DIRS})
    add_library( ${PROJECT_NAME} SHARED utils.cpp BaseONNX.cpp face_model_family.cpp )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OPENMP_INCLUDES})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS} OpenMP::OpenMP_CXX )
elseif(BUILD_PLATFORM STREQUAL "linux")
    add_library( ${PROJECT_NAME} SHARED utils.cpp BaseONNX.cpp face_model_family.cpp )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OPENMP_INCLUDES})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS} OpenMP::OpenMP_CXX )
elseif(BUILD_PLATFORM STREQUAL "macos")
    add_library( ${PROJECT_NAME} SHARED utils.cpp BaseONNX.cpp face_model_family.cpp )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OPENMP_INCLUDES})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS} OpenMP::OpenMP_CXX )
elseif(BUILD_PLATFORM STREQUAL "ios")
    add_library( ${PROJECT_NAME} STATIC utils.cpp BaseONNX.cpp face_model_family.cpp )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()






# if(BUILD_PLATFORM STREQUAL "linux")
#     target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
#     # target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX ${onnxruntime_LIBS})
#     # opencv protobuf conflicts with onnxruntime protobuf
#     # while DNN/ONNX: Build OpenCV with Protobuf to import ONNX models in function 'readNetFromONNX'
#     # so there can be only one module built in our engine between OPENCV::DNN and ONNX
#     target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX )
# elseif(BUILD_PLATFORM STREQUAL "android")
#     target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json ${OpenMP_omp_LIBRARY})
# elseif(BUILD_PLATFORM STREQUAL "macos")
#     target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES})
#     target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nlohmann_json::nlohmann_json OpenMP::OpenMP_CXX)
# endif()
# file(COPY $<TARGET_FILE:GustoEngine>
#      DESTINATION ${CMAKE_BINARY_DIR}/build/Android/${ANDROID_ABI})
# get_filename_component(targetfile "$<TARGET_FILE:GustoEngine>" NAME)

# message(${CMAKE_BINARY_DIR}/Android/${ANDROID_ABI}/${targetfile})

if(BUILD_PLATFORM STREQUAL "android")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/)
    add_custom_command(TARGET GustoEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:GustoEngine> ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/
    )

    # ----------------------------------------------------------------------------
    #   Install target, for "make install"
    # ----------------------------------------------------------------------------
    # install(FILES $<TARGET_FILE:GustoEngine> DESTINATION ${PROJECT_SOURCE_DIR}/../gusto-engine-unity-wrapper/Assets/Plugins/Android/)
    install(FILES $<TARGET_FILE:GustoEngine> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Android/${ANDROID_ABI})
elseif(BUILD_PLATFORM STREQUAL "linux")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Linux/)
    add_custom_command(TARGET GustoEngine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:GustoEngine> ${CMAKE_BINARY_DIR}/lib/Linux/
    )
    # install(FILES $<TARGET_FILE:GustoEngine> DESTINATION ${PROJECT_SOURCE_DIR}/../gusto-engine-unity-wrapper/Assets/Plugins/Linux/)
    install(FILES $<TARGET_FILE:GustoEngine> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Linux)
endif()

