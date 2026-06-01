add_executable(cxcu_example_device_query
    device_query.c
)
target_link_libraries(cxcu_example_device_query PRIVATE cxcu)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/kernels")
configure_file(
    kernels/vector_scale.cu
    "${CMAKE_CURRENT_BINARY_DIR}/kernels/vector_scale.cu"
    COPYONLY
)
configure_file(
    kernels/vector_math.h
    "${CMAKE_CURRENT_BINARY_DIR}/kernels/vector_math.h"
    COPYONLY
)

add_executable(cxcu_example_nvrtc_vector_scale
    nvrtc_vector_scale.c
)
target_link_libraries(cxcu_example_nvrtc_vector_scale PRIVATE cxcu)
target_compile_definitions(cxcu_example_nvrtc_vector_scale PRIVATE
    CXCU_EXAMPLE_KERNEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/kernels/vector_scale.cu"
)
