if(NOT BUILD_PLATFORM MATCHES "^(linux|macos|windows)$")
    message(STATUS "Skipping face_geometry_example on ${BUILD_PLATFORM}")
    return()
endif()

foreach(required_module IN ITEMS
        opencv_calib3d
        opencv_highgui
        opencv_imgcodecs
        opencv_videoio)
    if(NOT required_module IN_LIST OpenCV_LIBS)
        message(STATUS
            "Skipping face_geometry_example: ${required_module} is unavailable")
        return()
    endif()
endforeach()

add_executable(face_geometry_example
    face_geometry_example.cc
    ${SOURCES}
    ${FACE_MODEL_LIBS}
    ${NMS_LIBS}
    ${FACE_GEOMETRY_LIBS}
)

target_include_directories(face_geometry_example PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(face_geometry_example PRIVATE
    ${AI_ENGINE_LINK_LIBRARIES}
)

if(BUILD_PLATFORM STREQUAL "linux")
    target_compile_definitions(face_geometry_example PRIVATE DEBUG)
endif()
