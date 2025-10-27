
# pylib 
pybind11_add_module(libswd main.cpp pack.cpp init.cpp)
target_link_libraries(libswd PRIVATE aniso vti mesh shared ${LAPACKE_LIBRARIES} )
set_target_properties(aniso vti mesh shared  PROPERTIES POSITION_INDEPENDENT_CODE ON)
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