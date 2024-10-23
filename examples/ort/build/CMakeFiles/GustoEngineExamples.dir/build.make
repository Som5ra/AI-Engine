# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/build

# Include any dependencies generated for this target.
include CMakeFiles/GustoEngineExamples.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/GustoEngineExamples.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/GustoEngineExamples.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GustoEngineExamples.dir/flags.make

CMakeFiles/GustoEngineExamples.dir/codegen:
.PHONY : CMakeFiles/GustoEngineExamples.dir/codegen

CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.o: CMakeFiles/GustoEngineExamples.dir/flags.make
CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/ort_test.cpp
CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.o: CMakeFiles/GustoEngineExamples.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.o -MF CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.o.d -o CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/ort_test.cpp

CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/ort_test.cpp > CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.i

CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/ort_test.cpp -o CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.s

# Object files for target GustoEngineExamples
GustoEngineExamples_OBJECTS = \
"CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.o"

# External object files for target GustoEngineExamples
GustoEngineExamples_EXTERNAL_OBJECTS =

GustoEngineExamples: CMakeFiles/GustoEngineExamples.dir/ort_test.cpp.o
GustoEngineExamples: CMakeFiles/GustoEngineExamples.dir/build.make
GustoEngineExamples: /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/../../3rdparty/onnxruntime-static/lib/libonnxruntime.a
GustoEngineExamples: CMakeFiles/GustoEngineExamples.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable GustoEngineExamples"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GustoEngineExamples.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GustoEngineExamples.dir/build: GustoEngineExamples
.PHONY : CMakeFiles/GustoEngineExamples.dir/build

CMakeFiles/GustoEngineExamples.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GustoEngineExamples.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GustoEngineExamples.dir/clean

CMakeFiles/GustoEngineExamples.dir/depend:
	cd /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/build /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/build /media/sombrali/HDD1/opencv-unity/gusto_dnn/examples/build/CMakeFiles/GustoEngineExamples.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/GustoEngineExamples.dir/depend

