cmake_minimum_required(VERSION 3.18)  # Minimum required for good CUDA/C++17 support
project(SQLProcessor LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)

# Disable PIE to avoid illegal instruction issues on older CPUs
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")

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

# Link filesystem on older GCC (<9)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    string(REPLACE "." ";" GCC_VER_LIST ${GCC_VERSION})
    list(GET GCC_VER_LIST 0 GCC_VER_MAJOR)
    if (GCC_VER_MAJOR LESS 9)
        target_link_libraries(sql_processor PRIVATE stdc++fs)
    endif()
endif()

# Optional: custom 'run' target
add_custom_target(run
    COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sql_processor"
    DEPENDS sql_processor
    WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    COMMENT "Running SQL Processor..."
)
