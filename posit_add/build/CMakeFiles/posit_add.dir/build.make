# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

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
CMAKE_SOURCE_DIR = /home/amritha/Project/Operator/posit_add

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/amritha/Project/Operator/posit_add/build

# Include any dependencies generated for this target.
include CMakeFiles/posit_add.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/posit_add.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/posit_add.dir/flags.make

CMakeFiles/posit_add.dir/op.cpp.o: CMakeFiles/posit_add.dir/flags.make
CMakeFiles/posit_add.dir/op.cpp.o: ../op.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/amritha/Project/Operator/posit_add/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/posit_add.dir/op.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/posit_add.dir/op.cpp.o -c /home/amritha/Project/Operator/posit_add/op.cpp

CMakeFiles/posit_add.dir/op.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/posit_add.dir/op.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/amritha/Project/Operator/posit_add/op.cpp > CMakeFiles/posit_add.dir/op.cpp.i

CMakeFiles/posit_add.dir/op.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/posit_add.dir/op.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/amritha/Project/Operator/posit_add/op.cpp -o CMakeFiles/posit_add.dir/op.cpp.s

# Object files for target posit_add
posit_add_OBJECTS = \
"CMakeFiles/posit_add.dir/op.cpp.o"

# External object files for target posit_add
posit_add_EXTERNAL_OBJECTS =

libposit_add.so: CMakeFiles/posit_add.dir/op.cpp.o
libposit_add.so: CMakeFiles/posit_add.dir/build.make
libposit_add.so: /home/amritha/.local/lib/python3.8/site-packages/torch/lib/libtorch.so
libposit_add.so: /home/amritha/.local/lib/python3.8/site-packages/torch/lib/libc10.so
libposit_add.so: /home/amritha/.local/lib/python3.8/site-packages/torch/lib/libc10.so
libposit_add.so: CMakeFiles/posit_add.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/amritha/Project/Operator/posit_add/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libposit_add.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/posit_add.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/posit_add.dir/build: libposit_add.so

.PHONY : CMakeFiles/posit_add.dir/build

CMakeFiles/posit_add.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/posit_add.dir/cmake_clean.cmake
.PHONY : CMakeFiles/posit_add.dir/clean

CMakeFiles/posit_add.dir/depend:
	cd /home/amritha/Project/Operator/posit_add/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/amritha/Project/Operator/posit_add /home/amritha/Project/Operator/posit_add /home/amritha/Project/Operator/posit_add/build /home/amritha/Project/Operator/posit_add/build /home/amritha/Project/Operator/posit_add/build/CMakeFiles/posit_add.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/posit_add.dir/depend

