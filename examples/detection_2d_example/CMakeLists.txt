if(NOT BUILD_PLATFORM MATCHES "^(linux|macos)$")
    message(STATUS "Skipping detection_example on ${BUILD_PLATFORM}")
    return()
endif()

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
