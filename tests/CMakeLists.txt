set(UNIT_TESTS
    love_op
    rayl_op
    aniso_op
    att_aniso
    gqtable
)

foreach(TEST_NAME ${UNIT_TESTS})
    add_executable(test_${TEST_NAME} unit_tests/${TEST_NAME}.cpp)
    target_link_libraries(test_${TEST_NAME} aniso vti mesh shared
        ${LAPACKE_LIBRARIES} GTest::gtest_main)
    add_test(NAME test_${TEST_NAME} COMMAND test_${TEST_NAME})
endforeach()