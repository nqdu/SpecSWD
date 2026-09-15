
# pylib 
pybind11_add_module(libswd main.cpp pack.cpp init.cpp)
target_link_libraries(libswd PRIVATE aniso vti mesh shared ${LAPACKE_LIBRARIES} )
set_target_properties(aniso vti mesh shared  PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(SPECSWD_INSTALL_RPATH "")
if(LAPACKE_LIBRARIES)
  list(GET LAPACKE_LIBRARIES 0 SPECSWD_LAPACKE_LIBRARY)
  get_filename_component(SPECSWD_LAPACKE_LIBRARY_DIR
    "${SPECSWD_LAPACKE_LIBRARY}" DIRECTORY)
  list(APPEND SPECSWD_INSTALL_RPATH "${SPECSWD_LAPACKE_LIBRARY_DIR}")
endif()
foreach(SPECSWD_HDF5_LIBRARY IN LISTS HDF5_C_LIBRARIES)
  if(IS_ABSOLUTE "${SPECSWD_HDF5_LIBRARY}")
    get_filename_component(SPECSWD_HDF5_LIBRARY_DIR
      "${SPECSWD_HDF5_LIBRARY}" DIRECTORY)
    list(APPEND SPECSWD_INSTALL_RPATH "${SPECSWD_HDF5_LIBRARY_DIR}")
  endif()
endforeach()
list(REMOVE_DUPLICATES SPECSWD_INSTALL_RPATH)
set_target_properties(libswd PROPERTIES
  INSTALL_RPATH "${SPECSWD_INSTALL_RPATH}")
#target_include_directories(libswd  PUBLIC ${PROJECT_SOURCE_DIR}/src)
install(TARGETS libswd
  COMPONENT python
  LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/specd/lib/
)
install(TARGETS libswd
  COMPONENT python
  LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
)
 
# # .so 
# add_library(specswd SHARED pack.cpp)
# set_target_properties(specswd PROPERTIES PUBLIC_HEADER specswd.hpp )
# target_link_libraries(specswd PRIVATE vti mesh shared LAPACK::LAPACK -llapacke )
# install(TARGETS specswd
#     LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
#     PUBLIC_HEADER DESTINATION ${PROJECT_SOURCE_DIR}/include
# )
# add_library(libcps330 ${SOURCE})
