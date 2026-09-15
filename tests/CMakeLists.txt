set(UNIT_TESTS
    love_op
    rayl_op
    att_aniso
    gqtable
)

foreach(TEST_NAME ${UNIT_TESTS})
    add_executable(test_${TEST_NAME} unit_tests/${TEST_NAME}.cpp)
    target_link_libraries(test_${TEST_NAME} aniso vti mesh shared
        ${LAPACKE_LIBRARIES} GTest::gtest_main)
    add_test(NAME test_${TEST_NAME} COMMAND test_${TEST_NAME})
endforeach()

add_executable(test_aniso_op unit_tests/aniso_op.cpp)
target_link_libraries(test_aniso_op aniso vti mesh shared
    ${LAPACKE_LIBRARIES} GTest::gtest_main)

# Each anisotropic finite-difference case performs several QZ solves.  Keep
# them in separate processes because some HPC login environments impose a
# short per-process CPU limit.
foreach(ANISO_CASE
    CHECK_OP_CORRECTNESS
    ATTENUATING_GROUP_VELOCITY_MATCHES_FREQUENCY_DIFFERENCE
    COUPLED_ACOUSTIC_GROUP_VELOCITY_MATCHES_FREQUENCY_DIFFERENCE
    COUPLED_ACOUSTIC_GROUP_KERNEL_MATCHES_DIFFERENCE
    PHASE_KERNEL_MATCHES_POINTWISE_DIFFERENCE
    GROUP_KERNELS_MATCH_POINTWISE_DIFFERENCE
)
    add_test(
        NAME test_aniso_op_${ANISO_CASE}
        COMMAND test_aniso_op
            --gtest_filter=AnisoOperatorTest.${ANISO_CASE}
    )
endforeach()

add_executable(test_io unit_tests/io.cpp)
target_link_libraries(test_io shared GTest::gtest_main)
add_test(NAME test_io COMMAND test_io)

add_executable(io_cpp_bridge io_cpp_bridge.cpp)
target_link_libraries(io_cpp_bridge shared)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(
    NAME test_io_cross_language
    COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_io_cross_language.py
        $<TARGET_FILE:io_cpp_bridge>
        ${PROJECT_SOURCE_DIR}/specd/io.py
)
