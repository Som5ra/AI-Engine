cmake_minimum_required(VERSION 3.20.0)

project(GustoEngine)


file(GLOB FACE_GEOMETRY_LIBS ${CMAKE_SOURCE_DIR}/tools/face_geometry/*.cpp)
file(GLOB NMS_LIBS ${CMAKE_SOURCE_DIR}/tools/nms/*.cpp)


set(Engine_Required_Modules
    ${onnxruntime_LIBS}
    ${OpenCV_LIBS}
    ${GLOBAL_LINK_3RD_PARTY_LIBS}
)

if(BUILD_PLATFORM STREQUAL "android")
    include_directories(${onnxruntime_INCLUDE_DIRS})
    add_library( ${PROJECT_NAME} SHARED utils.cpp BaseONNX.cpp face_model_family.cpp ${FACE_GEOMETRY_LIBS} ${NMS_LIBS} )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OPENMP_INCLUDES})
    target_link_libraries(${PROJECT_NAME} ${Engine_Required_Modules} OpenMP::OpenMP_CXX )
elseif(BUILD_PLATFORM STREQUAL "linux")
    add_library( ${PROJECT_NAME} SHARED utils.cpp BaseONNX.cpp face_model_family.cpp ${FACE_GEOMETRY_LIBS} ${NMS_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OPENMP_INCLUDES})
    target_link_libraries(${PROJECT_NAME} ${Engine_Required_Modules} OpenMP::OpenMP_CXX )
elseif(BUILD_PLATFORM STREQUAL "macos")
    add_library(${PROJECT_NAME} SHARED utils.cpp BaseONNX.cpp face_model_family.cpp ${FACE_GEOMETRY_LIBS} ${NMS_LIBS} )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OPENMP_INCLUDES})
    target_link_libraries(${PROJECT_NAME} ${AppleLink} ${Engine_Required_Modules} OpenMP::OpenMP_CXX)
elseif(BUILD_PLATFORM STREQUAL "ios")
    add_library( ${PROJECT_NAME} STATIC utils.cpp BaseONNX.cpp face_model_family.cpp ${FACE_GEOMETRY_LIBS} ${NMS_LIBS} )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} "-framework Foundation" ${Engine_Required_Modules} ${GLOBAL_LINK_3RD_PARTY_LIBS} )


    # make a bundle
    # cannot auto-configure ${EFFECTIVE_PLATFORM_NAME} in $<TARGET_FILE:xxx>, so hardcode it here
    # pls see https://cmake.org/pipermail/cmake/2016-March/063050.html
    set(STATUS custom_target_path ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-"iphoneos"/${PROJECT_NAME}.a)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${libtool} -static -o libgusto_engine_fat.a ${custom_target_path} ${OpenCV_LIB_FILES} ${onnxruntime_LIBS}
        COMMENT "Bundling ${PROJECT_NAME}"
        VERBATIM)
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

