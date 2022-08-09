cmake_minimum_required(VERSION 3.15...3.20)

project(
  LearningCUDA
  VERSION 0.0.0
  LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

option(ENABLE_THRUST "Enable Thrust" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

# Handling build type nicely Set a default build type if none was specified
set(default_build_type "Release")
if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
  set(default_build_type "RelWithDebInfo")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(
    STATUS
      "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE
      "${default_build_type}"
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

# Include directories
add_subdirectory(examplesCUDA)

if(ENABLE_THRUST)
  find_package(Thrust REQUIRED CONFIG PATHS
               "/home/remi/lib/thrust/thrust/cmake")
  thrust_create_target(Thrust)
  add_subdirectory(examplesTHRUST)
endif(ENABLE_THRUST)
