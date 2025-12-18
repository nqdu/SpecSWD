file(GLOB SOURCE "*.cpp" )
list(FILTER SOURCE EXCLUDE REGEX "main.cpp")

add_library(aniso ${SOURCE})

add_executable(surfani main.cpp)
target_link_libraries(surfani aniso mesh shared ${LAPACKE_LIBRARIES})