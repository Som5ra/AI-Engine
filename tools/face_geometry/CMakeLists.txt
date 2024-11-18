cmake_minimum_required(VERSION 3.20.0)

project(face_geometry)
add_definitions("-DBUILD_PLATFORM_${BUILD_PLATFORM}")




include_directories(${CMAKE_SOURCE_DIR}/include)
if(BUILD_PLATFORM STREQUAL "ios")
    add_library( ${PROJECT_NAME} STATIC calculator.cpp face_geometry.cpp procrustes_solver.cpp geometry_pipeline.cpp )
    target_include_directories(${PROJECT_NAME} PRIVATE)
    target_link_libraries( ${PROJECT_NAME} ${GLOBAL_LINK_3RD_PARTY_LIBS})
elseif(BUILD_PLATFORM STREQUAL "windows")
    add_library( ${PROJECT_NAME} SHARED face_geometry.cpp)
        
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".dll")
    target_include_directories(${PROJECT_NAME} PRIVATE)
    target_link_libraries( ${PROJECT_NAME} ${GLOBAL_LINK_3RD_PARTY_LIBS})
else()
    include_directories(${OPENMP_INCLUDES})
    add_library( ${PROJECT_NAME} SHARED calculator.cpp face_geometry.cpp procrustes_solver.cpp geometry_pipeline.cpp )

    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} )
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()




# ----------------------------------------------------------------------------
#   Install target, for "make install"
# ----------------------------------------------------------------------------
if(BUILD_PLATFORM STREQUAL "android")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/)
    add_custom_command(TARGET face_geometry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:face_geometry> ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/
    )
    install(FILES $<TARGET_FILE:face_geometry> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Android/${ANDROID_ABI})
elseif(BUILD_PLATFORM STREQUAL "linux")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Linux/)
    add_custom_command(TARGET face_geometry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:face_geometry> ${CMAKE_BINARY_DIR}/lib/Linux/
    )
    install(FILES $<TARGET_FILE:face_geometry> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Linux)
elseif(BUILD_PLATFORM STREQUAL "windows")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Windows/)
    add_custom_command(TARGET face_geometry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:face_geometry> ${CMAKE_BINARY_DIR}/lib/Windows/
    )
    install(FILES $<TARGET_FILE:face_geometry> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Windows)
elseif(BUILD_PLATFORM STREQUAL "macos")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/MacOS/)
    add_custom_command(TARGET face_geometry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:face_geometry> ${CMAKE_BINARY_DIR}/lib/MacOS/
    )
    install(FILES $<TARGET_FILE:face_geometry> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/MacOS)
endif()