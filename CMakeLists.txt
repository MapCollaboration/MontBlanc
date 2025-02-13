# ============================
# BASIC DEFINITIONS
# ============================

# Define code version
set(MontBlanc_VERSION 4.8.0)

# Define minimum version of cmake
cmake_minimum_required(VERSION 3.10)

# Define project name and its language
project(MontBlanc CXX)

# Define c++ standard and issue all the warning demanded by this standard
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fPIC -O3")

if (NOT DEFINED CMAKE_MACOSX_RPATH)
   set(CMAKE_MACOSX_RPATH 0)
endif ()

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
   cmake_policy(SET CMP0148 OLD)
endif ()

# ============================
# MAIN
# ============================

# Define include directory
include_directories(${PROJECT_SOURCE_DIR}/inc)

# Generate list of source files
file(GLOB_RECURSE  source_files src/*)

# Define target library
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
add_library(MontBlanc SHARED ${source_files})

# Define libraries to be linked
target_link_libraries(MontBlanc)

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

target_link_libraries(MontBlanc ${YAML_LDFLAGS} ${EIGEN3_LDFLAGS} ${CERES_LIBRARIES} ${GLOG_LDFLAGS} ${GFLAGS_LDFLAGS} ${APFELXX_LIBRARIES} ${NANGAPARBAT_LIBRARIES} ${LHAPDF_LIBRARIES} ${GSL_LIBRARIES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${NNAD_CFLAGS} ${EIGEN3_CFLAGS} ${Ceres_CFLAGS} ${GLOG_CFLAGS} ${GFLAGS_CFLAGS} ${LHAPDF_CXX_FLAGS} ${APFELXX_CXX_FLAGS} ${NANGAPARBAT_CXX_FLAGS} ${GSL_CXX_FLAGS} ${YAML_CFLAGS} -Wall -fPIC -Wunused")

# Enable testing and build test codes
add_subdirectory(tests)
add_subdirectory(run)

# ============================
# CONFIGURATION
# ============================

# Configure version to file
configure_file("${PROJECT_SOURCE_DIR}/inc/MontBlanc/version.h.in" "${PROJECT_SOURCE_DIR}/inc/MontBlanc/version.h")

# Configure configuration script
set(prefix ${CMAKE_INSTALL_PREFIX})
set(includedir "${prefix}/include")
set(libdir "${prefix}/lib")
set(cflags "-I${prefix}/include")
set(ldflags "-L${prefix}/lib -lMontBlanc")
configure_file("${PROJECT_SOURCE_DIR}/bin/MontBlanc-config.in" "${PROJECT_BINARY_DIR}/bin/MontBlanc-config")

# ============================
# INSTALLATION
# ============================

# Installation of code
install(DIRECTORY inc/MontBlanc DESTINATION include)
install(FILES ${PROJECT_BINARY_DIR}/bin/MontBlanc-config DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
