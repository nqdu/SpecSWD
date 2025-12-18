#  CMakeLists.txt

cmake_minimum_required(VERSION 3.25.0)

################## compiler options ##################
# compiler
set(CMAKE_CXX_COMPILER          ${CXX})
set(CMAKE_Fortran_COMPILER          ${FC})

# compiler flags
set(CMAKE_CXX_FLAGS             ${CXXFLAGS}) 

################## dependencies ##################
# preferred installation prefix of dependencies
set(EIGEN_INC                 ${EIGEN_INC})

option(BUILD_LIBS "BUILD_LIBS" OFF)


# set default values
macro(setDefault var val)
if(NOT DEFINED ${var})
set(${var} ${val})
endif()
endmacro()
setDefault(EIGEN_INC   "~/software/eigen-3.4.0")

# project
project(SWDTTI LANGUAGES CXX Fortran)

# build type 
set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_SOURCE_DIR}/build/obj)

################# cmake setup #################
set(CMAKE_CXX_FLAGS  "-g -Wall -DEIGEN_DONT_PARALLELIZE")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
  set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -march=native -ffixed-line-length-none")
  #set(CMAKE_CXX_FLAGS_RELEASE  "-O0 -D_GLIBCXX_DEBUG")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -xHost -DEIGEN_STRONG_INLINE=inline")
  set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -xHost -extend-source")
else ()
  set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -march=native -ffixed-line-length-none")
  set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  #set(CMAKE_CXX_FLAGS_RELEASE  "-O0 -D_GLIBCXX_DEBUG")
endif()

set(CMAKE_VERBOSE_MAKEFILE on)

################# find packages #################
# eigen
include_directories(${EIGEN_INC})

# lapack
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(LAPACK REQUIRED)
find_package(LAPACKE REQUIRED)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DEIGEN_USE_LAPACKE")

#find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()

################# include paths #################
include_directories(${PROJECT_SOURCE_DIR}/src)
#include_directories(${PROJECT_SOURCE_DIR}/include)

# subs
add_subdirectory(${PROJECT_SOURCE_DIR}/src/shared)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/mesh)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/vti)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/aniso)
# add_subdirectory(${PROJECT_SOURCE_DIR}/src/multiphysics)

# check build libraries
if(BUILD_LIBS) 
  find_package(Python3 COMPONENTS Interpreter Development)
  find_package(pybind11 CONFIG)
  add_subdirectory(src/cps330)
  add_subdirectory(src/libswd)
endif()
