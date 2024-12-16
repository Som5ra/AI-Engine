cmake_minimum_required(VERSION 3.20.0)

project(nms)
add_definitions("-DBUILD_PLATFORM_${BUILD_PLATFORM}")


include_directories(${CMAKE_SOURCE_DIR}/include)
if(BUILD_PLATFORM STREQUAL "ios")
    add_library( ${PROJECT_NAME} STATIC nms.cc)
    target_include_directories(${PROJECT_NAME} PRIVATE )
    target_link_libraries( ${PROJECT_NAME} nlohmann_json::nlohmann_json)
elseif(BUILD_PLATFORM STREQUAL "windows")
    add_library( ${PROJECT_NAME} SHARED nms.cc)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".dll")
    target_include_directories(${PROJECT_NAME} PRIVATE )
    target_link_libraries( ${PROJECT_NAME} nlohmann_json::nlohmann_json)
else()
    include_directories(${OPENMP_INCLUDES})
    add_library( ${PROJECT_NAME} SHARED nms.cc)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES})
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX nlohmann_json::nlohmann_json)
endif()


# ----------------------------------------------------------------------------
#   Install target, for "make install"
# ----------------------------------------------------------------------------
if(BUILD_PLATFORM STREQUAL "android")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/)
    add_custom_command(TARGET nms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nms> ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/
    )
    install(FILES $<TARGET_FILE:nms> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Android/${ANDROID_ABI})
elseif(BUILD_PLATFORM STREQUAL "linux")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Linux/)
    add_custom_command(TARGET nms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nms> ${CMAKE_BINARY_DIR}/lib/Linux/
    )
    install(FILES $<TARGET_FILE:nms> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Linux)
elseif(BUILD_PLATFORM STREQUAL "windows")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Windows/)
    add_custom_command(TARGET nms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nms> ${CMAKE_BINARY_DIR}/lib/Windows/
    )
    install(FILES $<TARGET_FILE:nms> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Windows)
elseif(BUILD_PLATFORM STREQUAL "macos")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/MacOS/)
    add_custom_command(TARGET nms POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:nms> ${CMAKE_BINARY_DIR}/lib/MacOS/
    )
    install(FILES $<TARGET_FILE:nms> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/MacOS)
endif()