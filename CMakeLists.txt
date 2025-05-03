cmake_minimum_required(VERSION 3.18)
project(SQLProcessor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# Include headers
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/sql-parser-main/src
    ${PROJECT_SOURCE_DIR}/libs/linenoise-ng-master/include
)

# (If needed: Make sure find_library finds your local sqlparser.a, or set SQLPARSER_LIB manually)
link_directories(
    ${PROJECT_SOURCE_DIR}/sql-parser-main/build
)

# Gather sources
file(GLOB_RECURSE CPP_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/libs/linenoise-ng-master/src/*.cpp
)
file(GLOB_RECURSE CU_SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cu
)

# Set output dir for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Define executable, .cpp and .cu together
add_executable(sql_processor ${CPP_SOURCES} ${CU_SOURCES})

# CUDA requirements for proper linking:
set_target_properties(sql_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80" # (or set for your GPU: 52 for Maxwell, 60/61 for Pascal, etc.)
)

target_compile_options(sql_processor PRIVATE
    -Wall -Wextra
)

# Modern CMake: Prefer imported CUDA::cudart target if available, else fallback
find_package(CUDAToolkit)
if (CUDAToolkit_FOUND AND TARGET CUDA::cudart)
    target_link_libraries(sql_processor
        CUDA::cudart
        sqlparser
    )
else()
    target_link_libraries(sql_processor
        cudart
        sqlparser
    )
endif()