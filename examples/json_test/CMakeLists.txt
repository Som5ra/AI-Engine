cmake_minimum_required(VERSION 3.20.0)

project(json_test)


include_directories(${CMAKE_SOURCE_DIR}/include)

if(BUILD_PLATFORM STREQUAL "windows")

    add_executable( ${PROJECT_NAME} SHARED json_test.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp)
    
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".dll")
    target_include_directories(${PROJECT_NAME} PRIVATE)
    target_link_libraries( ${PROJECT_NAME} )
elseif(BUILD_PLATFORM STREQUAL "ios")
    add_library( ${PROJECT_NAME} STATIC json_test.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE)
    target_link_libraries( ${PROJECT_NAME} ${GLOBAL_LINK_3RD_PARTY_LIBS})
else()
    # enable omp.h
    include_directories(${OPENMP_INCLUDES})

    add_executable( ${PROJECT_NAME} json_test.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} )
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX ${GLOBAL_LINK_3RD_PARTY_LIBS})
endif()


