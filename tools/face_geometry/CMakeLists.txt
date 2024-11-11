cmake_minimum_required(VERSION 3.20.0)

project(face_geometry)
add_definitions("-DBUILD_PLATFORM_${BUILD_PLATFORM}")
if(BUILD_PLATFORM STREQUAL "android")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g0")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    if (ANDROID_ABI STREQUAL "arm64-v8a")
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/aarch64/libomp.a")
    elseif (ANDROID_ABI STREQUAL "armeabi-v7a")
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/arm/libomp.a")
    elseif(ANDROID_ABI STREQUAL "x86")
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/i386/libomp.a")
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(OpenMP_omp_LIBRARY "/home/sombrali/Unity/Hub/Editor/2022.3.20f1/Editor/Data/PlaybackEngines/AndroidPlayer/NDK/toolchains/llvm/prebuilt/linux-x86_64/lib64/clang/12.0.8/lib/linux/x86_64/libomp.a")
    endif()

endif()

if(BUILD_PLATFORM STREQUAL "linux")
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if(BUILD_PLATFORM STREQUAL "macos")
    set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/llvm/19.1.3/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm/19.1.3/bin/clang++")
    set(OPENMP_LIBRARIES "/opt/homebrew/Cellar/llvm/19.1.3/lib")
    set(OPENMP_INCLUDES "/opt/homebrew/Cellar/llvm/19.1.3/include")
    
    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "omp")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
    endif()
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.a")
endif()

if(BUILD_PLATFORM STREQUAL "ios")
    # set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/llvm/19.1.3/bin/clang")
    # set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm/19.1.3/bin/clang++")
    set(OPENMP_LIBRARIES "/opt/homebrew/Cellar/llvm/19.1.3/lib")
    set(OPENMP_INCLUDES "/opt/homebrew/Cellar/llvm/19.1.3/include")
    # enable_language(CXX)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON) #Set this to ON if you want to use GNU++
    set(CMAKE_CXX_STANDARD 17)
    # message(STATUS "[SOMBRA2] -> ${CMAKE_CXX_COMPILER}")
    # if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
    #     set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    #     set(OpenMP_C_LIB_NAMES "omp")
    # endif()

    # if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
    #     set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    #     set(OpenMP_CXX_LIB_NAMES "omp")
    # endif()
    # set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/llvm/lib/libomp.a")
endif()

if(BUILD_PLATFORM STREQUAL "windows")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    set(CMAKE_SYSTEM_NAME Windows)
    set(TOOLCHAIN_PREFIX x86_64-w64-mingw32)
    set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc)
    set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
    set(CMAKE_Fortran_COMPILER ${TOOLCHAIN_PREFIX}-gfortran)
    set(CMAKE_RC_COMPILER ${TOOLCHAIN_PREFIX}-windres)

    set(CMAKE_FIND_ROOT_PATH /usr/${TOOLCHAIN_PREFIX})

    # modify default behavior of FIND_XXX() commands
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

    # set(OpenMP_omp_LIBRARY "/usr/x86_64-w64-mingw32/lib/libdcomp.a")
    add_library( ${PROJECT_NAME} SHARED face_geometry.cpp)
    
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".dll")
    target_include_directories(${PROJECT_NAME} PRIVATE)
    target_link_libraries( ${PROJECT_NAME} )
else()
    find_package (Eigen3 REQUIRED)    
    find_package(OpenMP REQUIRED)
    # enable omp.h
    if(BUILD_PLATFORM STREQUAL "macos")
        # set(OpenMP_OMPH "/opt/homebrew/opt/libomp/include")
        include_directories(${OPENMP_INCLUDES})
        message(STATUS ${OPENMP_INCLUDES})
    endif()

    include_directories(${CMAKE_SOURCE_DIR}/include)
    message(STATUS "Include: ${CMAKE_SOURCE_DIR}/include")
    # find_package(nlohmann_json REQUIRED)
    # add_executable( ${PROJECT_NAME} json_test.cpp ${CMAKE_SOURCE_DIR}/src/utils.cpp)
    if (BUILD_PLATFORM STREQUAL "ios")
        add_library( ${PROJECT_NAME} STATIC calculator.cpp face_geometry.cpp procrustes_solver.cpp geometry_pipeline.cpp )
    else()
        add_library( ${PROJECT_NAME} SHARED calculator.cpp face_geometry.cpp procrustes_solver.cpp geometry_pipeline.cpp )
    endif()

    target_include_directories(${PROJECT_NAME} PRIVATE ${OPENMP_INCLUDES} )
    target_link_libraries( ${PROJECT_NAME} OpenMP::OpenMP_CXX Eigen3::Eigen nlohmann_json::nlohmann_json)
endif()



# ----------------------------------------------------------------------------
#   Install target, for "make install"
# ----------------------------------------------------------------------------
if(BUILD_PLATFORM STREQUAL "android")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/)
    add_custom_command(TARGET face_geometry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:face_geometry> ${CMAKE_BINARY_DIR}/../lib/Android/${ANDROID_ABI}/
    )
    install(FILES $<TARGET_FILE:face_geometry> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Android/${ANDROID_ABI})
elseif(BUILD_PLATFORM STREQUAL "linux")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Linux/)
    add_custom_command(TARGET face_geometry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:face_geometry> ${CMAKE_BINARY_DIR}/lib/Linux/
    )
    install(FILES $<TARGET_FILE:face_geometry> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Linux)
elseif(BUILD_PLATFORM STREQUAL "windows")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/Windows/)
    add_custom_command(TARGET face_geometry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:face_geometry> ${CMAKE_BINARY_DIR}/lib/Windows/
    )
    install(FILES $<TARGET_FILE:face_geometry> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/Windows)
elseif(BUILD_PLATFORM STREQUAL "macos")
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib/MacOS/)
    add_custom_command(TARGET face_geometry POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:face_geometry> ${CMAKE_BINARY_DIR}/lib/MacOS/
    )
    install(FILES $<TARGET_FILE:face_geometry> DESTINATION /media/sombrali/HDD1/opencv-unity/AI-Engine-Unity-Example/Assets/Plugins/MacOS)
endif()