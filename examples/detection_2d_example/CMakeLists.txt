if(NOT BUILD_PLATFORM MATCHES "^(linux|macos)$")
    message(STATUS "Skipping detection_example on ${BUILD_PLATFORM}")
    return()
endif()

foreach(required_module IN ITEMS
        opencv_highgui
        opencv_imgcodecs
        opencv_videoio)
    if(NOT required_module IN_LIST OpenCV_LIBS)
        message(STATUS
            "Skipping detection_example: ${required_module} is unavailable")
        return()
    endif()
endforeach()

add_executable(detection_example
    detection_example.cc
    ${SOURCES}
    ${DETECTION_2D_LIBS}
    ${NMS_LIBS}
)

target_include_directories(detection_example PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(detection_example PRIVATE
    ${AI_ENGINE_LINK_LIBRARIES}
)

if(BUILD_PLATFORM STREQUAL "linux")
    target_compile_definitions(detection_example PRIVATE DEBUG)
endif()
