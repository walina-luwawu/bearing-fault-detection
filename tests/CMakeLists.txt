find_package(GTest CONFIG REQUIRED)

add_executable(
    unitTests
    testMatrix.cpp
)

target_link_libraries(
    unitTests
    PRIVATE 
    fault-detection-core
    GTest::gtest
    GTest::gmock
    GTest::gmock_main
)

include(GoogleTest)
gtest_discover_tests(unitTests)