cmake_minimum_required (VERSION 3.5)
# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if (NOT DEFINED CMAKE_MACOSX_RPATH)
   set(CMAKE_MACOSX_RPATH 0)
endif()

# the project name
project(MontBlanc)

# activating some global properties for the project
set(MontBlanc_VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# export version to file
configure_file(
  "${PROJECT_SOURCE_DIR}/inc/MontBlanc/version.h.in"
  "${PROJECT_SOURCE_DIR}/inc/MontBlanc/version.h"
)

find_package(PkgConfig REQUIRED)
pkg_search_module(YAML REQUIRED yaml-cpp)
pkg_search_module(EIGEN3 REQUIRED eigen3)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
pkg_search_module(GLOG REQUIRED libglog)
pkg_search_module(GFLAGS REQUIRED gflags)

# Configuration script
set(prefix ${CMAKE_INSTALL_PREFIX})
set(exec_prefix "${prefix}")
set(includedir "${prefix}/include")
set(libdir "${prefix}/lib")
configure_file(
  "${PROJECT_SOURCE_DIR}/bin/MontBlanc-config.in"
  "${PROJECT_SOURCE_DIR}/bin/MontBlanc-config"
)

include_directories(${PROJECT_SOURCE_DIR}/inc)

# LHAPDF
find_program(LHAPDF_CONFIG lhapdf-config REQUIRED)
if (LHAPDF_CONFIG)
   execute_process(COMMAND ${LHAPDF_CONFIG} --cflags OUTPUT_VARIABLE LHAPDF_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process(COMMAND ${LHAPDF_CONFIG} --libs OUTPUT_VARIABLE LHAPDF_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)
   set(LHAPDF_CXX_FLAGS ${LHAPDF_CXX_FLAGS} CACHE STRING INTERNAL)
   set(LHAPDF_LIBRARIES ${LHAPDF_LIBRARIES} CACHE STRING INTERNAL)
else(LHAPDF_CONFIG)
  message(FATAL_ERROR "LHAPDF not found!")
endif(LHAPDF_CONFIG)

# APFEL++
find_program(APFELXX_CONFIG apfelxx-config REQUIRED)
if (APFELXX_CONFIG)
   execute_process(COMMAND ${APFELXX_CONFIG} --cppflags OUTPUT_VARIABLE APFELXX_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process(COMMAND ${APFELXX_CONFIG} --ldflags OUTPUT_VARIABLE APFELXX_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)
   set(APFELXX_CXX_FLAGS ${APFELXX_CXX_FLAGS} CACHE STRING INTERNAL)
   set(APFELXX_LIBRARIES ${APFELXX_LIBRARIES} CACHE STRING INTERNAL)
else(APFELXX_CONFIG)
  message(FATAL_ERROR "APFEL++ not found!")
endif(APFELXX_CONFIG)

# NANGAPARBAT
find_program(NANGAPARBAT_CONFIG NangaParbat-config REQUIRED)
if (NANGAPARBAT_CONFIG)
   execute_process(COMMAND ${NANGAPARBAT_CONFIG} --cppflags OUTPUT_VARIABLE NANGAPARBAT_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process(COMMAND ${NANGAPARBAT_CONFIG} --ldflags OUTPUT_VARIABLE NANGAPARBAT_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)
   set(NANGAPARBAT_CXX_FLAGS ${NANGAPARBAT_CXX_FLAGS} CACHE STRING INTERNAL)
   set(NANGAPARBAT_LIBRARIES ${NANGAPARBAT_LIBRARIES} CACHE STRING INTERNAL)
else(NANGAPARBAT_CONFIG)
  message(FATAL_ERROR "NangaParbat not found!")
endif(NANGAPARBAT_CONFIG)

# GSL
find_program(GSL_CONFIG gsl-config)
if (GSL_CONFIG)
   execute_process(COMMAND ${GSL_CONFIG} --cflags OUTPUT_VARIABLE GSL_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
   execute_process(COMMAND ${GSL_CONFIG} --libs OUTPUT_VARIABLE GSL_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)
   set(GSL_CXX_FLAGS ${GSL_CXX_FLAGS} CACHE STRING INTERNAL)
   set(GSL_LIBRARIES ${GSL_LIBRARIES} CACHE STRING INTERNAL)
else(GSL_CONFIG)
  message ("GSL not found. Some parts of the code will not be compiled.")
endif(GSL_CONFIG)

# NNAD
find_program(NNAD_CONFIG nnad-config REQUIRED)
if (NNAD_CONFIG)
   execute_process(COMMAND ${NNAD_CONFIG} --cflags OUTPUT_VARIABLE NNAD_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
   set(NNAD_CFLAGS ${NNAD_CFLAGS} CACHE STRING INTERNAL)
endif(NNAD_CONFIG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NNAD_CFLAGS} ${EIGEN3_CFLAGS} ${Ceres_CFLAGS} ${GLOG_CFLAGS} ${GFLAGS_CFLAGS} ${LHAPDF_CXX_FLAGS} ${APFELXX_CXX_FLAGS} ${NANGAPARBAT_CXX_FLAGS} ${GSL_CXX_FLAGS} ${YAML_CFLAGS} -Wall -fPIC -Wunused")

# build target library
add_subdirectory(src)
add_subdirectory(run)
add_subdirectory(tests)

#option(BUILD_TESTS "Build tests code" ON)
#if(BUILD_TESTS)
#  enable_testing()
#  add_subdirectory(tests)
#endif(BUILD_TESTS)

install(FILES ${PROJECT_SOURCE_DIR}/bin/MontBlanc-config
	   DESTINATION bin
	   PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
