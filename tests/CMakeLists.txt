add_executable(test_cxcu
    cxcu.test.c
)
target_link_libraries(test_cxcu PRIVATE cxcu)
add_test(NAME cxcu COMMAND "$<TARGET_FILE:test_cxcu>")

add_executable(test_cxcu_grid_bench
    cxcu.grid_bench.test.c
)
target_link_libraries(test_cxcu_grid_bench PRIVATE cxcu)
add_test(NAME cxcu_grid_bench COMMAND "$<TARGET_FILE:test_cxcu_grid_bench>" --smoke)

add_executable(test_cxcu_batched_parameter_sweep
    cxcu.batched_parameter_sweep.test.c
)
target_link_libraries(test_cxcu_batched_parameter_sweep PRIVATE cxcu)
add_test(
    NAME cxcu_batched_parameter_sweep
    COMMAND "$<TARGET_FILE:test_cxcu_batched_parameter_sweep>" 256 128
)

add_executable(test_cxcu_device_view_bench
    cxcu.device_view_bench.test.c
)
target_link_libraries(test_cxcu_device_view_bench PRIVATE cxcu m)
target_compile_options(test_cxcu_device_view_bench PRIVATE -UNDEBUG)
target_compile_definitions(test_cxcu_device_view_bench PRIVATE
    CXCU_TEST_INCLUDE_OPTION="-I${CMAKE_CURRENT_SOURCE_DIR}/../include")
add_test(
    NAME cxcu_device_view_bench_smoke
    COMMAND "$<TARGET_FILE:test_cxcu_device_view_bench>" 4096 3
)
