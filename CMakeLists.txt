cmake_minimum_required(VERSION 3.16)
project(llama_shmem LANGUAGES CXX)

find_program(OSHCXX_COMPILER NAMES oshc++ oshcxx)
if(OSHCXX_COMPILER)
    set(CMAKE_CXX_COMPILER "${OSHCXX_COMPILER}" CACHE STRING "OpenSHMEM C++ compiler" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_RPATH "/usr/local/lib")

add_compile_options(-O3 -march=native -Wall -Wextra -Wno-unused-parameter)

find_path(LLAMA_INCLUDE_DIR NAMES llama.h PATHS /usr/local/include REQUIRED)
find_library(LLAMA_LIB NAMES llama PATHS /usr/local/lib REQUIRED)
find_library(GGML_LIB NAMES ggml PATHS /usr/local/lib REQUIRED)
find_library(GGML_BASE_LIB NAMES ggml-base PATHS /usr/local/lib)
find_library(GGML_CPU_LIB NAMES ggml-cpu PATHS /usr/local/lib)

set(GGML_LIBS ${LLAMA_LIB} ${GGML_LIB})
if(GGML_BASE_LIB)
    list(APPEND GGML_LIBS ${GGML_BASE_LIB})
endif()
if(GGML_CPU_LIB)
    list(APPEND GGML_LIBS ${GGML_CPU_LIB})
endif()

add_executable(llama_shmem src/llama_shmem.cpp)
target_include_directories(llama_shmem PRIVATE ${LLAMA_INCLUDE_DIR})
target_link_libraries(llama_shmem PRIVATE ${GGML_LIBS} pthread m)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/test_shmem_pipe.cpp")
    add_executable(test_shmem_pipe src/test_shmem_pipe.cpp)
    target_link_libraries(test_shmem_pipe PRIVATE pthread m)
endif()
