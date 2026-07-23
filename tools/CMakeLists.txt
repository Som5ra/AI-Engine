if(AI_ENGINE_BUILD_STANDALONE_TOOLS)
    add_subdirectory(nms)
    add_subdirectory(face_geometry)
endif()

if(AI_ENGINE_BUILD_6D_TRACKING)
    add_subdirectory(6d_tracking)
endif()
