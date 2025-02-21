cmake_minimum_required(VERSION 3.20.0)

project(GustoEngineWASM)


set(Engine_Required_Modules
    # base modules: 
    ${SOURCES}
    # models:
    ${FACE_MODEL_LIBS}
    ${HUMAN_SEG_LIBS}
    ${HUMAN_POSE_LIBS}
    ${DETECTION_2D_LIBS}
    # tools(post-processing):
    ${FACE_GEOMETRY_LIBS} 
    ${NMS_LIBS}
)

set(Engine_Required_Link_Modules
    ${onnxruntime_LIBS}
    ${OpenCV_LIBS}
    ${GLOBAL_LINK_3RD_PARTY_LIBS}
)

file(GLOB WASM_API_SOURCES
    ${CMAKE_SOURCE_DIR}/src_wasm_api/*.cc
)

message("WASM_API_SOURCES: ${WASM_API_SOURCES}")

# if(BUILD_PLATFORM STREQUAL "wasm")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1")
# set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=256MB -s EXIT_RUNTIME=1")

    # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -sEXPORTED_FUNCTIONS=[_add] -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue','lengthBytesUTF8','UTF8ToString']")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -sEXPORTED_FUNCTIONS=[_add] -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue','lengthBytesUTF8','UTF8ToString']")
    # set(CMAKE_EXECUTBLE_LINKER_FLAGS "${CMAKE_EXECUTBLE_LINKER_FLAGS} -sEXPORTED_FUNCTIONS=[_add] -sEXPORTED_RUNTIME_METHODS=['ccall','stringToUTF8','setValue','getValue','lengthBytesUTF8','UTF8ToString']")

    # add_executable( ${PROJECT_NAME} ${WASM_API_SOURCES} ${Engine_Required_Modules})
add_definitions(-std=c++17)
set(CMAKE_CXX_STANDARD 17)

add_library( gusto_onnx SHARED ${Engine_Required_Modules})
target_include_directories(gusto_onnx PRIVATE ${onnxruntime_INCLUDE_DIRS} )
target_link_libraries(gusto_onnx ${Engine_Required_Link_Modules} )

add_executable( ${PROJECT_NAME} ${WASM_API_SOURCES})
# add_executable( ${PROJECT_NAME} wasm_api.cc)
target_include_directories(${PROJECT_NAME} PRIVATE ${onnxruntime_INCLUDE_DIRS} )
target_link_libraries(${PROJECT_NAME} gusto_onnx )

set(CUSTOM_FLAGS " -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=512MB -s ALLOW_MEMORY_GROWTH=1 ")
# set(CUSTOM_FLAGS " -s FORCE_FILESYSTEM=1 -s INITIAL_MEMORY=512MB")
string(APPEND CUSTOM_FLAGS " -fopenmp ")
string(APPEND CUSTOM_FLAGS " -sSTACK_SIZE=10485760 -sEXPORTED_FUNCTIONS=['_malloc','_free']")
string(APPEND CUSTOM_FLAGS " --bind -s ERROR_ON_UNDEFINED_SYMBOLS=0 -Os --preload-file ${CMAKE_SOURCE_DIR}/src_wasm_api/assets@/")

set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS ${CUSTOM_FLAGS})

# set_target_properties(${PROJECT_NAME}  PROPERTIES LINK_FLAGS "--bind -s ERROR_ON_UNDEFINED_SYMBOLS=0 --js-library  ${CMAKE_SOURCE_DIR}/src_wasm_api/library.js -O3")
# set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-s SIDE_MODULE=1 ")
# set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS    "-s WASM=1")
# endif()

