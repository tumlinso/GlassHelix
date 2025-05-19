cmake_minimum_required(VERSION 3.29)

# Directory configuration: APACHE-ARROW AND TORCH MUST BE IN PATH!

set(CUDA_DIR "/usr/local/cuda")
option(USE_CUDA "Enable CUDA support" ON)

project(GlassHelix LANGUAGES CXX)

# Use clang++ and C++23
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_STANDARD    23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS  OFF)

# Project-wide: use libc++ and embed rpath so the loader finds it at runtime
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

# Set source directory and include directories
set(CMAKE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

find_package(OpenMP REQUIRED)

# Configure CUDA
if (USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_COMPILER       "${CUDA_DIR}/bin/nvcc")
  set(CMAKE_CUDA_ARCHITECTURES  86)
  set(CMAKE_CUDA_FLAGS          "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  set(CMAKE_CUDA_HOST_COMPILER  "${CMAKE_CXX_COMPILER}")

  set(CMAKE_CUDA_STANDARD       20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS     OFF)
  find_package(CUDAToolkit REQUIRED)

  set(USE_CUDNN      ON)
  set(USE_CUFILE     ON)
  set(USE_CUDSS      ON)
  set(USE_CUSPARSELT ON)
else()
  message(STATUS "Building without CUDA support")
  set(USE_CUDNN      OFF)
  set(USE_CUFILE     OFF)
  set(USE_CUDSS      OFF)
  set(USE_CUSPARSELT OFF)
endif()

# Configure Torch (Caffe2 dependencies)
if (USE_CUDA)
  set(CAFFE2_USE_CUDNN      ON)
  set(CAFFE2_USE_CUFILE     ON)
  set(CAFFE2_USE_CUDSS      ON)
  set(CAFFE2_USE_CUSPARSELT ON)
else()
  set(CAFFE2_USE_CUDNN      OFF)
  set(CAFFE2_USE_CUFILE     OFF)
  set(CAFFE2_USE_CUDSS      OFF)
  set(CAFFE2_USE_CUSPARSELT OFF)
endif()
find_package(Torch REQUIRED)

# Configure Arrow
find_package(Arrow REQUIRED)
if (USE_CUDA)
  set(ARROW_USE_CUDA ON)
  find_package(ArrowCUDA REQUIRED)
else()
  set(ARROW_USE_CUDA OFF)
endif()

# -------------------------------------------------------------------
# GlassHelix (main) executable
# -------------------------------------------------------------------
add_executable(GlassHelix
        src/Cell.hh
        src/Feature.hh
        src/Sequence.hh
        src/Dataset.hh
        src/Block.hh
        src/incomplete/Genome.hh
)
target_sources(
        GlassHelix
        PRIVATE
        FILE_SET cxx_modules TYPE CXX_MODULES
        FILES
)
target_link_libraries(GlassHelix
        "${TORCH_LIBRARIES}"
        "${ARROW_LIBRARIES}"
        "${ARROW_CUDA_LIBRARIES}"
        "${CUDAToolkit_LIBRARIES}"
        OpenMP::OpenMP_CXX
)

# -------------------------------------------------------------------
# BlockTest executable
# -------------------------------------------------------------------
add_executable(BlockTest
        src/tests/BlockTest.cpp
        src/Block.hh
)
target_compile_options(BlockTest PRIVATE -O3)
target_link_libraries(BlockTest PRIVATE OpenMP::OpenMP_CXX)

# -------------------------------------------------------------------
# BlockPerformanceTest executable
# -------------------------------------------------------------------
add_executable(BlockPerformanceTest
        src/tests/BlockPerformanceTest.cpp
        src/Block.hh
)
target_compile_options(BlockPerformanceTest PRIVATE -O3)
target_link_libraries(BlockPerformanceTest PRIVATE OpenMP::OpenMP_CXX)

# -------------------------------------------------------------------
# DictionaryTest executable
# -------------------------------------------------------------------
add_executable(DictionaryTest
        src/tests/DictionaryTest.cpp
)
target_compile_options(DictionaryTest PRIVATE -O3)
target_link_libraries(DictionaryTest)

# -------------------------------------------------------------------
# SequenceTest executable
# -------------------------------------------------------------------
add_executable(SequenceTest
        src/tests/SequenceTest.cpp
        src/Sequence.hh
)
target_compile_options(SequenceTest PRIVATE -O3)
target_link_libraries(SequenceTest PRIVATE OpenMP::OpenMP_CXX)
