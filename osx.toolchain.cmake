# set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/llvm@18/18.1.8/bin/clang")
# set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/llvm@18/18.1.8/bin/clang++")
# set(OPENMP_LIBRARIES "/opt/homebrew/Cellar/llvm@18/18.1.8/lib")
# set(OPENMP_INCLUDES "/opt/homebrew/Cellar/llvm@18/18.1.8/include")


if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
endif()



# set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/19.1.5/lib/libomp.a")



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fobjc-arc")
set(AppleLink "-framework Foundation")