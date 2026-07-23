if(NOT BUILD_PLATFORM MATCHES "^(linux|android|macos|windows)$")
    message(FATAL_ERROR
        "Experimental 6D tracking is not supported on ${BUILD_PLATFORM}")
endif()

if(BUILD_PLATFORM STREQUAL "android")
    set(AI_ENGINE_6D_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/body.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/camera.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/common.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/model.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/region_modality.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/tracker.cpp"
    )
    set(AI_ENGINE_6D_GRAPHICS_LIBRARIES "")
else()
    find_package(OpenGL REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(glfw3 3 REQUIRED)
    file(GLOB_RECURSE AI_ENGINE_6D_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    )
    set(AI_ENGINE_6D_GRAPHICS_LIBRARIES
        OpenGL::GL
        GLEW::GLEW
        glfw
    )
endif()

add_library(ai_engine_6d_objects OBJECT
    ${AI_ENGINE_6D_SOURCES}
    "${CMAKE_CURRENT_SOURCE_DIR}/csrt3d/csrt3d.cc"
)
set_target_properties(ai_engine_6d_objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(ai_engine_6d_objects PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrt3d"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
)
target_link_libraries(ai_engine_6d_objects PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    ${AI_ENGINE_OPENMP_TARGET}
    ${AI_ENGINE_6D_GRAPHICS_LIBRARIES}
)

if(BUILD_PLATFORM STREQUAL "android")
    target_compile_definitions(ai_engine_6d_objects PUBLIC __DISABLE_OPENGL__)
else()
    target_compile_definitions(ai_engine_6d_objects PUBLIC __OPENCV_DEBUG__)
endif()

foreach(target_name IN ITEMS csrt3d_module model_target)
    add_library(${target_name} SHARED
        "${CMAKE_CURRENT_SOURCE_DIR}/csrt3d/unity_api.cc"
    )
    target_link_libraries(${target_name} PRIVATE ai_engine_6d_objects)
endforeach()

if(NOT BUILD_PLATFORM STREQUAL "android")
    add_executable(run_on_webcam
        "${CMAKE_CURRENT_SOURCE_DIR}/csrt3d/example/run_on_webcam.cc"
    )
    target_link_libraries(run_on_webcam PRIVATE ai_engine_6d_objects)
endif()

install(TARGETS csrt3d_module model_target
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
