if(BUILD_PLATFORM STREQUAL "ios")
    set(FACE_GEOMETRY_LIBRARY_TYPE STATIC)
else()
    set(FACE_GEOMETRY_LIBRARY_TYPE SHARED)
endif()

add_library(face_geometry ${FACE_GEOMETRY_LIBRARY_TYPE}
    calculator.cc
    face_geometry.cc
    procrustes_solver.cc
    geometry_pipeline.cc
)

target_include_directories(face_geometry PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(face_geometry PRIVATE
    ${GLOBAL_LINK_3RD_PARTY_LIBS}
    ${AI_ENGINE_OPENMP_TARGET}
)

install(TARGETS face_geometry
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
