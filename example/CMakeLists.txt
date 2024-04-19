cmake_minimum_required(VERSION 3.0)

add_executable(benchmark benchmark.cpp)
set_property(TARGET benchmark PROPERTY CXX_STANDARD 11)
target_link_libraries(benchmark PRIVATE flatkdtree::flatkdtree)

add_executable(integrity integrity.cpp)
set_property(TARGET integrity PROPERTY CXX_STANDARD 11)
target_link_libraries(integrity PRIVATE flatkdtree::flatkdtree)
