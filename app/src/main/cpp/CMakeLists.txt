cmake_minimum_required(VERSION 3.22.1)

project("leantype_llama" C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

# llama.cpp (git submodule, pinned to a release tag): the library and the common utils
# (chat templates of the models, sampling), nothing that downloads or starts processes
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
# ggml's own thread pool instead of OpenMP
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)

if(DEFINED ANDROID_ABI AND ANDROID_ABI STREQUAL "arm64-v8a")
    # fast kernels for quantized models on ARM (i8mm, SME), chosen at runtime
    set(GGML_CPU_KLEIDIAI ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(llama.cpp build-llama)

add_library(${CMAKE_PROJECT_NAME} SHARED leantype_llama.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME} llama llama-common)
if(ANDROID)
    target_link_libraries(${CMAKE_PROJECT_NAME} android log)
endif()
