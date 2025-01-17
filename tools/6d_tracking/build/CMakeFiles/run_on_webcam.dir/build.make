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
CMAKE_COMMAND = /home/sombrali/cmake-3.30.6-linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/sombrali/cmake-3.30.6-linux-x86_64/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build

# Include any dependencies generated for this target.
include CMakeFiles/run_on_webcam.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/run_on_webcam.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/run_on_webcam.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/run_on_webcam.dir/flags.make

CMakeFiles/run_on_webcam.dir/src/body.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/body.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/body.cpp
CMakeFiles/run_on_webcam.dir/src/body.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/run_on_webcam.dir/src/body.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/body.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/body.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/body.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/body.cpp

CMakeFiles/run_on_webcam.dir/src/body.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/body.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/body.cpp > CMakeFiles/run_on_webcam.dir/src/body.cpp.i

CMakeFiles/run_on_webcam.dir/src/body.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/body.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/body.cpp -o CMakeFiles/run_on_webcam.dir/src/body.cpp.s

CMakeFiles/run_on_webcam.dir/src/camera.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/camera.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/camera.cpp
CMakeFiles/run_on_webcam.dir/src/camera.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/run_on_webcam.dir/src/camera.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/camera.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/camera.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/camera.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/camera.cpp

CMakeFiles/run_on_webcam.dir/src/camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/camera.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/camera.cpp > CMakeFiles/run_on_webcam.dir/src/camera.cpp.i

CMakeFiles/run_on_webcam.dir/src/camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/camera.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/camera.cpp -o CMakeFiles/run_on_webcam.dir/src/camera.cpp.s

CMakeFiles/run_on_webcam.dir/src/common.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/common.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/common.cpp
CMakeFiles/run_on_webcam.dir/src/common.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/run_on_webcam.dir/src/common.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/common.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/common.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/common.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/common.cpp

CMakeFiles/run_on_webcam.dir/src/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/common.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/common.cpp > CMakeFiles/run_on_webcam.dir/src/common.cpp.i

CMakeFiles/run_on_webcam.dir/src/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/common.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/common.cpp -o CMakeFiles/run_on_webcam.dir/src/common.cpp.s

CMakeFiles/run_on_webcam.dir/src/model.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/model.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/model.cpp
CMakeFiles/run_on_webcam.dir/src/model.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/run_on_webcam.dir/src/model.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/model.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/model.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/model.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/model.cpp

CMakeFiles/run_on_webcam.dir/src/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/model.cpp > CMakeFiles/run_on_webcam.dir/src/model.cpp.i

CMakeFiles/run_on_webcam.dir/src/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/model.cpp -o CMakeFiles/run_on_webcam.dir/src/model.cpp.s

CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/normal_renderer.cpp
CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/normal_renderer.cpp

CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/normal_renderer.cpp > CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.i

CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/normal_renderer.cpp -o CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.s

CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/normal_viewer.cpp
CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/normal_viewer.cpp

CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/normal_viewer.cpp > CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.i

CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/normal_viewer.cpp -o CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.s

CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/occlusion_renderer.cpp
CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/occlusion_renderer.cpp

CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/occlusion_renderer.cpp > CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.i

CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/occlusion_renderer.cpp -o CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.s

CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/region_modality.cpp
CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/region_modality.cpp

CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/region_modality.cpp > CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.i

CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/region_modality.cpp -o CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.s

CMakeFiles/run_on_webcam.dir/src/renderer.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/renderer.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/renderer.cpp
CMakeFiles/run_on_webcam.dir/src/renderer.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/run_on_webcam.dir/src/renderer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/renderer.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/renderer.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/renderer.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/renderer.cpp

CMakeFiles/run_on_webcam.dir/src/renderer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/renderer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/renderer.cpp > CMakeFiles/run_on_webcam.dir/src/renderer.cpp.i

CMakeFiles/run_on_webcam.dir/src/renderer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/renderer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/renderer.cpp -o CMakeFiles/run_on_webcam.dir/src/renderer.cpp.s

CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/renderer_geometry.cpp
CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/renderer_geometry.cpp

CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/renderer_geometry.cpp > CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.i

CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/renderer_geometry.cpp -o CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.s

CMakeFiles/run_on_webcam.dir/src/tracker.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/tracker.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/tracker.cpp
CMakeFiles/run_on_webcam.dir/src/tracker.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/run_on_webcam.dir/src/tracker.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/tracker.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/tracker.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/tracker.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/tracker.cpp

CMakeFiles/run_on_webcam.dir/src/tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/tracker.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/tracker.cpp > CMakeFiles/run_on_webcam.dir/src/tracker.cpp.i

CMakeFiles/run_on_webcam.dir/src/tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/tracker.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/tracker.cpp -o CMakeFiles/run_on_webcam.dir/src/tracker.cpp.s

CMakeFiles/run_on_webcam.dir/src/viewer.cpp.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/src/viewer.cpp.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/viewer.cpp
CMakeFiles/run_on_webcam.dir/src/viewer.cpp.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/run_on_webcam.dir/src/viewer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/src/viewer.cpp.o -MF CMakeFiles/run_on_webcam.dir/src/viewer.cpp.o.d -o CMakeFiles/run_on_webcam.dir/src/viewer.cpp.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/viewer.cpp

CMakeFiles/run_on_webcam.dir/src/viewer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/src/viewer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/viewer.cpp > CMakeFiles/run_on_webcam.dir/src/viewer.cpp.i

CMakeFiles/run_on_webcam.dir/src/viewer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/src/viewer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/src/viewer.cpp -o CMakeFiles/run_on_webcam.dir/src/viewer.cpp.s

CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/csrt3d/csrt3d.cc
CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.o -MF CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.o.d -o CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/csrt3d/csrt3d.cc

CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/csrt3d/csrt3d.cc > CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.i

CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/csrt3d/csrt3d.cc -o CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.s

CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.o: CMakeFiles/run_on_webcam.dir/flags.make
CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.o: /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/csrt3d/example/run_on_webcam.cc
CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.o: CMakeFiles/run_on_webcam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.o -MF CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.o.d -o CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.o -c /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/csrt3d/example/run_on_webcam.cc

CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/csrt3d/example/run_on_webcam.cc > CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.i

CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/csrt3d/example/run_on_webcam.cc -o CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.s

# Object files for target run_on_webcam
run_on_webcam_OBJECTS = \
"CMakeFiles/run_on_webcam.dir/src/body.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/camera.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/common.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/model.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/renderer.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/tracker.cpp.o" \
"CMakeFiles/run_on_webcam.dir/src/viewer.cpp.o" \
"CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.o" \
"CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.o"

# External object files for target run_on_webcam
run_on_webcam_EXTERNAL_OBJECTS =

run_on_webcam: CMakeFiles/run_on_webcam.dir/src/body.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/camera.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/common.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/model.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/normal_renderer.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/normal_viewer.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/occlusion_renderer.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/region_modality.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/renderer.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/renderer_geometry.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/tracker.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/src/viewer.cpp.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/csrt3d/csrt3d.cc.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/csrt3d/example/run_on_webcam.cc.o
run_on_webcam: CMakeFiles/run_on_webcam.dir/build.make
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libGL.so
run_on_webcam: /usr/lib/x86_64-linux-gnu/libGLEW.so
run_on_webcam: /usr/lib/x86_64-linux-gnu/libglfw.so.3.3
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
run_on_webcam: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
run_on_webcam: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
run_on_webcam: /usr/lib/x86_64-linux-gnu/libpthread.a
run_on_webcam: CMakeFiles/run_on_webcam.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Linking CXX executable run_on_webcam"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run_on_webcam.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/run_on_webcam.dir/build: run_on_webcam
.PHONY : CMakeFiles/run_on_webcam.dir/build

CMakeFiles/run_on_webcam.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/run_on_webcam.dir/cmake_clean.cmake
.PHONY : CMakeFiles/run_on_webcam.dir/clean

CMakeFiles/run_on_webcam.dir/depend:
	cd /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build /media/sombrali/HDD1/opencv-unity/gusto_dnn/tools/6d_tracking/build/CMakeFiles/run_on_webcam.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/run_on_webcam.dir/depend

