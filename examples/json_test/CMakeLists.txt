set(JSON_TEST_SOURCES
    json_test.cc
    "${CMAKE_SOURCE_DIR}/src/utils.cc"
)

if(BUILD_PLATFORM STREQUAL "ios")
    add_library(json_test STATIC ${JSON_TEST_SOURCES})
else()
    add_executable(json_test ${JSON_TEST_SOURCES})
endif()

target_include_directories(json_test PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(json_test PRIVATE
    ${GLOBAL_LINK_3RD_PARTY_LIBS}
    ${AI_ENGINE_OPENMP_TARGET}
)
