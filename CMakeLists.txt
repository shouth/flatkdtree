cmake_minimum_required(VERSION 3.0)
project(flatkdtree)

option(flatkdtree_BUILD_EXAMPLE "Build flatkdtree example" OFF)

add_library(flatkdtree INTERFACE)
target_include_directories(flatkdtree INTERFACE include)
target_compile_features(flatkdtree INTERFACE cxx_std_11)
add_library(flatkdtree::flatkdtree ALIAS flatkdtree)

if(flatkdtree_BUILD_EXAMPLE)
  add_subdirectory(example)
endif()
