if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "BUILD_PLATFORM=wasm requires the Emscripten CMake toolchain")
endif()

add_executable(GustoEngineWASM
    wasm_api.cc
    ${AI_ENGINE_MODULES}
)

target_include_directories(GustoEngineWASM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(GustoEngineWASM PRIVATE ${AI_ENGINE_LINK_LIBRARIES})

target_link_options(GustoEngineWASM PRIVATE
    "--bind"
    "-Os"
    "-sFORCE_FILESYSTEM=1"
    "-sINITIAL_MEMORY=536870912"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sSTACK_SIZE=10485760"
    "-sERROR_ON_UNDEFINED_SYMBOLS=1"
    "SHELL:-sEXPORTED_FUNCTIONS=['_malloc','_free']"
)

set_target_properties(GustoEngineWASM PROPERTIES SUFFIX ".js")
