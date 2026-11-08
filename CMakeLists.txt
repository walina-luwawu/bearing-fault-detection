cmake_minimum_required(VERSION 3.15)
project(BEARING_FAULT_DETECTION LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenBLAS CONFIG REQUIRED)

add_library(
    softFault
    src/matrix.cpp
    src/dataset.cpp
    src/softmaxRegression.cpp
)

target_include_directories(
    softFault
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    softFault
    PUBLIC OpenBLAS::OpenBLAS
)

option(BUILD_TESTS "Build unit tests" ON)

if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()