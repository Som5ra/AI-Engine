# set(CMAKE_SYSTEM_NAME Windows)
# set(TOOLCHAIN_PREFIX x86_64-w64-mingw32)
# set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}-gcc)
# set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}-g++)
# set(CMAKE_Fortran_COMPILER ${TOOLCHAIN_PREFIX}-gfortran)
# set(CMAKE_RC_COMPILER ${TOOLCHAIN_PREFIX}-windres)

# set(CMAKE_FIND_ROOT_PATH /usr/${TOOLCHAIN_PREFIX})

# # modify default behavior of FIND_XXX() commands
# set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
# set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


add_compile_options(
    $<$<CONFIG:>:/MT> #---------|
    $<$<CONFIG:Debug>:/MTd> #---|-- Statically link the runtime libraries
    $<$<CONFIG:Release>:/MT> #--|
    $<$<CONFIG:RelWithDebInfo>:/MT>
    $<$<CONFIG:MinSizeRel>:/MT>
)


add_definitions(-DOpenCV_STATIC=ON)

add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
add_definitions(-DCMAKE_BUILD_TYPE=Release)
