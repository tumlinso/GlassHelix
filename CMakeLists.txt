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

include_directories(.)

find_package(OpenMP REQUIRED)

# Configure CUDA
if (USE_CUDA)
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
# BlockTest executable
# -------------------------------------------------------------------
add_executable(BlockTest
        src/data/tests/BlockTest.cpp
        src/data/Block.hh
)
target_compile_options(BlockTest PRIVATE -O3)
target_link_libraries(BlockTest PRIVATE OpenMP::OpenMP_CXX)

# -------------------------------------------------------------------
# BlockPerformanceTest executable
# -------------------------------------------------------------------
add_executable(BlockPerformanceTest
        src/data/tests/BlockPerformanceTest.cpp
        src/data/Block.hh
)
target_compile_options(BlockPerformanceTest PRIVATE -O3)
target_link_libraries(BlockPerformanceTest PRIVATE OpenMP::OpenMP_CXX)

# -------------------------------------------------------------------
# DictionaryTest executable
# -------------------------------------------------------------------
add_executable(DictionaryTest
        src/data/tests/DictionaryTest.cpp
)
target_compile_options(DictionaryTest PRIVATE -O3)
target_include_directories(DictionaryTest PRIVATE
        ${CMAKE_SOURCE_DIR}/src/data
)
target_link_libraries(DictionaryTest)

# -------------------------------------------------------------------
# GlassHelix (main) executable
# -------------------------------------------------------------------
add_executable(GlassHelix
        src/main.cpp
        src/nn/transformer/virtual/Transformer.cppm
        src/nn/transformer/virtual/Tokenizer.cppm
        src/GeneTokenizer.cppm
        src/nn/transformer/virtual/Embedder.cppm
        src/BioLogic/Sequence.cppm
        src/nn/primitives/Linear.cppm
        src/nn/primitives/Linear.cppm
        src/nn/primitives/Perceptron.cppm
        src/nn/primitives/Perceptron.cppm
        src/nn/primitives/torch/TorchAttention.cppm
        src/nn/primitives/torch/TorchAttention.cppm
        src/Graph/Graph.cc
        src/Graph/Graph.hh
        src/Graph/Graph.inl
        archive/SparseMatrix.hh
        src/data/Dictionary.hh
        archive/BidirectionalMap.hh
        src/nn/primitives/Attention.hh
        src/nn/primitives/Attention.cc
        src/GlassHelix.cc
        src/GlassHelix.hh
        src/Cell.cc
        src/Cell.hh
        src/Cell.inl
        src/BioLogic/Feature.cc
        src/BioLogic/Feature.hh
        src/BioLogic/Feature.inl
        src/BioLogic/Sequence.cc
        src/BioLogic/Sequence.hh
        src/BioLogic/Sequence.inl
        src/data/Dataset.cc
        src/data/Dataset.hh
        src/data/Block.hh
        src/data/tests/BlockTest.cpp
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
