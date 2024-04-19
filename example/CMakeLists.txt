cmake_minimum_required(VERSION 3.0)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE flatkdtree::flatkdtree)

add_executable(integrity integrity.cpp)
target_link_libraries(integrity PRIVATE flatkdtree::flatkdtree)
