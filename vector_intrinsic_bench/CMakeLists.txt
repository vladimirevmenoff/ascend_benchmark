cmake_minimum_required(VERSION 3.16)
project(add_bench LANGUAGES CXX)

# CANN toolkit path — set via -DCMAKE_ASCEND_TOOLKIT_PATH=...
if(NOT DEFINED CMAKE_ASCEND_TOOLKIT_PATH)
    set(CMAKE_ASCEND_TOOLKIT_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

# SOC version — set via -DASCEND_SOC_VERSION=...
if(NOT DEFINED ASCEND_SOC_VERSION)
    set(ASCEND_SOC_VERSION "Ascend310P3")
endif()

set(ASCENDC_COMPILER "${CMAKE_ASCEND_TOOLKIT_PATH}/compiler/tikcpp/tikcpp_compiler")

# Kernel sources
set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernel/add_bench_kernel.cpp)

# Compile AscendC kernel
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/add_bench_kernel.o
    COMMAND ${ASCENDC_COMPILER}
        ${KERNEL_SRC}
        -I${CMAKE_CURRENT_SOURCE_DIR}/kernel
        --soc-version=${ASCEND_SOC_VERSION}
        -o ${CMAKE_CURRENT_BINARY_DIR}/add_bench_kernel.o
    DEPENDS ${KERNEL_SRC}
            ${CMAKE_CURRENT_SOURCE_DIR}/kernel/add_bench_kernel.h
    COMMENT "Compiling AscendC kernel for ${ASCEND_SOC_VERSION}"
)

add_custom_target(add_bench_kernel ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/add_bench_kernel.o
)
