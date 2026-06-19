cmake_minimum_required(VERSION 3.20)
project(cxcu VERSION 0.1.0 LANGUAGES C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CXCU_ENABLE_CUDA "Enable CUDA Driver API support" ON)
option(CXCU_BUILD_TESTS "Build cxcu tests" ${PROJECT_IS_TOP_LEVEL})
option(CXCU_BUILD_EXAMPLES "Build cxcu examples" ${PROJECT_IS_TOP_LEVEL})

set(CXCU_HAVE_CUDA OFF)
if(CXCU_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND TARGET CUDA::cuda_driver)
        set(CXCU_HAVE_CUDA ON)
    else()
        message(STATUS "CUDA Driver API not found. Building cxcu unavailable stub.")
    endif()
endif()

add_library(cxcu STATIC
    src/cxcu.c
)
add_library(cxcu::cxcu ALIAS cxcu)

target_include_directories(cxcu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(CXCU_HAVE_CUDA)
    target_compile_definitions(cxcu PRIVATE CXCU_HAVE_CUDA=1)
    target_link_libraries(cxcu PRIVATE CUDA::cuda_driver)
    if(TARGET CUDA::nvrtc)
        target_compile_definitions(cxcu PRIVATE CXCU_HAVE_NVRTC=1)
        target_link_libraries(cxcu PRIVATE CUDA::nvrtc)
    else()
        message(STATUS "CUDA NVRTC not found. Runtime CUDA source compilation disabled.")
    endif()
endif()

install(
    TARGETS cxcu
    EXPORT cxcuTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(
    EXPORT cxcuTargets
    FILE cxcuTargets.cmake
    NAMESPACE cxcu::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cxcu
)
configure_package_config_file(
    cmake/cxcuConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/cxcuConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cxcu
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cxcuConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cxcuConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cxcuConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cxcu
)

if(CXCU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CXCU_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
