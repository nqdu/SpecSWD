# macOS supplies BLAS/LAPACK, but not the LAPACKE entry points used by Eigen
# and the solver. Keep those calls unchanged with a local Accelerate adapter.
option(SPECSWD_USE_ACCELERATE "Use Apple Accelerate instead of external BLAS/LAPACK" ${APPLE})
# Remove the obsolete legacy/modern switch from existing build caches.
unset(SPECSWD_ACCELERATE_NEW_LAPACK CACHE)

if(SPECSWD_USE_ACCELERATE AND APPLE)
  set(_specswd_macos_target "${CMAKE_OSX_DEPLOYMENT_TARGET}")
  if(NOT _specswd_macos_target)
    execute_process(COMMAND /usr/bin/sw_vers -productVersion
      OUTPUT_VARIABLE _specswd_macos_target OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
  endif()
  if(_specswd_macos_target VERSION_LESS 13.3)
    message(STATUS "Accelerate's modern LAPACK requires macOS 13.3+; using external LAPACK/LAPACKE")
    set(SPECSWD_USE_ACCELERATE OFF CACHE BOOL
      "Use Apple Accelerate instead of external BLAS/LAPACK" FORCE)
  endif()
endif()

if(SPECSWD_USE_ACCELERATE)
  if(NOT APPLE)
    message(FATAL_ERROR "SPECSWD_USE_ACCELERATE requires an Apple platform")
  endif()
  find_library(SPECSWD_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  add_library(specswd_accelerate_lapacke STATIC
    ${PROJECT_SOURCE_DIR}/src/platform/accelerate_lapacke.cpp)
  target_include_directories(specswd_accelerate_lapacke PRIVATE ${EIGEN_INC})
  target_link_libraries(specswd_accelerate_lapacke PUBLIC ${SPECSWD_ACCELERATE_FRAMEWORK})
  set_target_properties(specswd_accelerate_lapacke PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # Apple's header requires this definition to expose the modern declarations
  # and bind calls to the $NEWLAPACK symbols. There is no legacy adapter path.
  target_compile_definitions(specswd_accelerate_lapacke PRIVATE ACCELERATE_NEW_LAPACK)
  set(SPECSWD_LAPACK_BACKEND "Apple Accelerate (modern, 32-bit LAPACK integers)" CACHE INTERNAL "Selected LAPACK backend" FORCE)
  set(LAPACK_LIBRARIES ${SPECSWD_ACCELERATE_FRAMEWORK})
  set(LAPACKE_LIBRARIES specswd_accelerate_lapacke)
else()
  find_package(LAPACK REQUIRED)
  find_package(LAPACKE REQUIRED)
  set(SPECSWD_LAPACK_BACKEND "External LAPACK/LAPACKE: ${LAPACKE_LIBRARIES}" CACHE INTERNAL "Selected LAPACK backend" FORCE)
endif()
message(STATUS "SpecSWD BLAS/LAPACK: ${SPECSWD_LAPACK_BACKEND}")
