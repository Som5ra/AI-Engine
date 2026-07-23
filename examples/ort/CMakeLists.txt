if(NOT BUILD_PLATFORM MATCHES "^(linux|macos|windows)$")
    message(STATUS "Skipping ort_test on ${BUILD_PLATFORM}")
    return()
endif()

add_executable(ort_test ort_test.cpp)
target_link_libraries(ort_test PRIVATE ai_engine_onnxruntime)
