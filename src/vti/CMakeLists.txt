file(GLOB SOURCE "*.cpp" )
list(FILTER SOURCE EXCLUDE REGEX "main_love.cpp")
list(FILTER SOURCE EXCLUDE REGEX "main_rayl.cpp")

add_library(vti ${SOURCE})

add_executable(surflove main_love.cpp)
add_executable(surfrayl main_rayl.cpp)
target_link_libraries(surflove vti mesh shared ${LAPACKE_LIBRARIES})
target_link_libraries(surfrayl vti mesh shared ${LAPACKE_LIBRARIES})