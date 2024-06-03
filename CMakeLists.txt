cmake_minimum_required(VERSION 3.20)

project(proj)

add_library(flags INTERFACE)

target_compile_features(flags INTERFACE cxx_std_20)

target_compile_options(flags INTERFACE -Wall;-Wextra;-O0;-mavx;-mavx2;)

target_include_directories(flags INTERFACE ${PROJECT_SOURCE_DIR}/include)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE flags)