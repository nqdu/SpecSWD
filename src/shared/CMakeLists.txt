file(GLOB SOURCE "*.cpp")

add_library(shared ${SOURCE})
target_include_directories(shared PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(shared PUBLIC ${HDF5_C_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/io.hpp
        DESTINATION include/shared)
