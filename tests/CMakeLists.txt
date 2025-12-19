add_executable(unit_tests
    unit_tests/test_love_op.cpp 
    unit_tests/test_rayl_op.cpp
    unit_tests/test_aniso_op.cpp
    unit_tests/test_att_aniso.cpp 
)

target_link_libraries(
    unit_tests  aniso vti mesh shared 
    ${LAPACKE_LIBRARIES} GTest::gtest_main
)

add_test(
    NAME unit_tests
    COMMAND unit_tests
)