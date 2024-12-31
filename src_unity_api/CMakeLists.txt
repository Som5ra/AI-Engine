cmake_minimum_required(VERSION 3.20.0)

project(GustoEngineUnity)


set(Engine_Required_Modules
    # base modules: 
    ${SOURCES}
    # models:
    ${FACE_MODEL_LIBS}
    ${HUMAN_SEG_LIBS}
    ${HUMAN_POSE_LIBS}
    ${DETECTION_2D_LIBS}
    # tools(post-processing):
    ${FACE_GEOMETRY_LIBS} 
    ${NMS_LIBS}
)

set(Engine_Required_Link_Modules
    ${onnxruntime_LIBS}
    ${OpenCV_LIBS}
    ${GLOBAL_LINK_3RD_PARTY_LIBS}
)

file(GLOB UNITY_API_SOURCES
    ${CMAKE_SOURCE_DIR}/src_unity_api/*.cc
)

if(BUILD_PLATFORM STREQUAL "android")
    include_directories(${onnxruntime_INCLUDE_DIRS})
    add_library( ${PROJECT_NAME} SHARED ${UNITY_API_SOURCES} ${Engine_Required_Modules} )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OPENMP_INCLUDES})
    target_link_libraries(${PROJECT_NAME} ${Engine_Required_Link_Modules} OpenMP::OpenMP_CXX )
elseif(BUILD_PLATFORM STREQUAL "linux")

    add_library( ${PROJECT_NAME} SHARED ${UNITY_API_SOURCES} ${Engine_Required_Modules})
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OPENMP_INCLUDES})
    target_link_libraries(${PROJECT_NAME} ${Engine_Required_Link_Modules} OpenMP::OpenMP_CXX )
elseif(BUILD_PLATFORM STREQUAL "macos")
    add_library(${PROJECT_NAME} SHARED ${UNITY_API_SOURCES} ${Engine_Required_Modules} )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OPENMP_INCLUDES})
    target_link_libraries(${PROJECT_NAME} ${AppleLink} ${Engine_Required_Link_Modules} OpenMP::OpenMP_CXX)
elseif(BUILD_PLATFORM STREQUAL "ios")
    add_library( ${PROJECT_NAME} STATIC ${UNITY_API_SOURCES} ${Engine_Required_Modules} )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} "-framework Foundation" ${Engine_Required_Link_Modules} ${GLOBAL_LINK_3RD_PARTY_LIBS} )

    # [Sombra] -> make a bundle here
    # cannot auto-configure ${EFFECTIVE_PLATFORM_NAME} in $<TARGET_FILE:xxx>, so hardcode it here
    # pls see https://cmake.org/pipermail/cmake/2016-March/063050.html
    set(custom_target_path ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}-iphoneos/lib${PROJECT_NAME}.a)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND libtool -static -o libgusto_engine_unity_fat.a ${custom_target_path} ${OpenCV_LIB_FILES} ${onnxruntime_LIBS}
        COMMENT "Bundling ${PROJECT_NAME}"
        VERBATIM)
elseif(BUILD_PLATFORM STREQUAL "windows")
    add_library(${PROJECT_NAME} SHARED ${UNITY_API_SOURCES} ${Engine_Required_Modules} )
    target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} ${OPENMP_INCLUDES})
    # target_link_libraries(${PROJECT_NAME} ${AppleLink} ${Engine_Required_Link_Modules} OpenMP::OpenMP_CXX)
    target_link_libraries(${PROJECT_NAME} ${Engine_Required_Link_Modules})
endif()



if(BUILD_PLATFORM STREQUAL "android")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/)
    add_custom_command(TARGET GustoEngineUnity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:GustoEngineUnity> ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/
    )

    # ----------------------------------------------------------------------------
    #   Install target, for "make install"
    # ----------------------------------------------------------------------------
    # install(FILES $<TARGET_FILE:GustoEngineUnity> DESTINATION ${PROJECT_SOURCE_DIR}/../gusto-engine-unity-wrapper/Assets/Plugins/Android/)
    install(FILES $<TARGET_FILE:GustoEngineUnity> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Android/${ANDROID_ABI})
elseif(BUILD_PLATFORM STREQUAL "linux")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Linux/)
    add_custom_command(TARGET GustoEngineUnity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:GustoEngineUnity> ${CMAKE_BINARY_DIR}/lib/Linux/
    )
    # install(FILES $<TARGET_FILE:GustoEngineUnity> DESTINATION ${PROJECT_SOURCE_DIR}/../gusto-engine-unity-wrapper/Assets/Plugins/Linux/)
    install(FILES $<TARGET_FILE:GustoEngineUnity> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Linux)
endif()

