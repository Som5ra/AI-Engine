if(BUILD_PLATFORM STREQUAL "ios")
    set(NMS_LIBRARY_TYPE STATIC)
else()
    set(NMS_LIBRARY_TYPE SHARED)
endif()

add_library(nms ${NMS_LIBRARY_TYPE} nms.cc)
target_include_directories(nms PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(nms PRIVATE
    nlohmann_json::nlohmann_json
    ${AI_ENGINE_OPENMP_TARGET}
)

install(TARGETS nms
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
