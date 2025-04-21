cmake_minimum_required(VERSION 3.18)  # Minimum required for good CUDA/C++17 support
project(SQLProcessor LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/sql-parser-main/src
)

# SQL Parser and CUDA libraries
link_directories(
    ${PROJECT_SOURCE_DIR}/sql-parser-main/build
    /usr/local/cuda/lib64
)

# Collect all .cpp and .cu source files
file(GLOB_RECURSE CPP_SOURCES
    src/*.cpp
)

file(GLOB_RECURSE CU_SOURCES
    src/*.cu
)

# Output binary directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Define the target
add_executable(sql_processor ${CPP_SOURCES} ${CU_SOURCES})

# Compile flags
target_compile_options(sql_processor PRIVATE
    -Wall -Wextra
)

# Link libraries
target_link_libraries(sql_processor
    sqlparser
    cudart
)
