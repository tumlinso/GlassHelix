cmake_minimum_required(VERSION 3.29)

# ─────────────────────────────────────────────────────────────────────────────
# Project-wide configuration
# ─────────────────────────────────────────────────────────────────────────────
project(GlassHelix LANGUAGES CXX)

set(CUDA_DIR "/usr/local/cuda")
option(USE_CUDA "Enable CUDA support" ON)

# Export compile_commands.json for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use clang++ and C++23
set(CMAKE_CXX_COMPILER       "clang++")
set(CMAKE_CXX_STANDARD       20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS     OFF)

# libc++ + embed rpath so the loader finds it at runtime
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

# OpenMP support
find_package(OpenMP REQUIRED COMPONENTS CXX)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA configuration
# ─────────────────────────────────────────────────────────────────────────────
if (USE_CUDA)
  set(CMAKE_CUDA_FLAGS          "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
  set(CMAKE_CUDA_COMPILER       "${CUDA_DIR}/bin/nvcc")
  set(CMAKE_CUDA_HOST_COMPILER  "${CMAKE_CXX_COMPILER}")
  set(CMAKE_CUDA_ARCHITECTURES  86)   # Ampere/Hopper
  set(CMAKE_CUDA_STANDARD       20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS     OFF)
  enable_language(CUDA)

  find_package(CUDAToolkit REQUIRED)
else()
  message(STATUS "Building without CUDA support")
endif()


# ─────────────────────────────────────────────────────────────────────────────
# Global include dirs
# ─────────────────────────────────────────────────────────────────────────────
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/models/GlassEtch/include
)


# ─────────────────────────────────────────────────────────────────────────────
# GlassHelix (main) executable
# ─────────────────────────────────────────────────────────────────────────────
add_executable(GlassHelix
        src/Cell.hh
        src/memory/Feature.hh
        src/Sequence.hh
        src/memory/Dataset.hh
        src/memory/AsynchronousArray.hh
        src/incomplete/Genome.hh

        src/models/GlassEtch/BinaryAutoencoder.hh
        src/models/GlassEtch/Binarizer.hh
        src/models/GlassEtch/Binarizer.cuh

        src/memory/SparseMatrix/CompressedSparseRow.hh
        src/memory/SparseMatrix/IRandomAccess.hh
        src/memory/SparseMatrix/CoordinateList.hh
        src/incomplete/Layers/SparseReader.cuh

        src/include/DeviceMacros.cuh
        src/TranscriptomeEncoder.hh

        src/models/GlassEtch/src/GlassEtch.cuh
        src/models/GlassEtch/src/main.cc
        src/include/DeviceMatrix.cuh
        src/tests/network/network.hh
        src/include/sigmoid.hh
        src/include/macros.h
        src/tests/network/FeedbackLoop.hh
        src/tests/network/NetworkLogicAND.hh
        src/tests/network/ORGate.hh
        src/tests/network/NetworkLogic.hh
        src/tests/network/Netowrl.cc
        src/tests/network/Network.inl
)

# Ensure any .cc including .cuh is compiled by NVCC
set_source_files_properties(
        src/models/GlassEtch/src/main.cc
        PROPERTIES LANGUAGE CUDA
)

target_link_libraries(GlassHelix
        PRIVATE
        $<$<BOOL:${USE_CUDA}>:${CUDAToolkit_LIBRARIES}>
        OpenMP::OpenMP_CXX
)


# ─────────────────────────────────────────────────────────────────────────────
# GlassEtch (minimal prototype) executable
# ─────────────────────────────────────────────────────────────────────────────
add_executable(GlassEtch
        src/models/GlassEtch/src/main.cc

        src/memory/AsynchronousArray.hh
        src/memory/SparseMatrix/CompressedSparseRow.hh
        src/memory/SparseMatrix/IRandomAccess.hh
        src/memory/SparseMatrix/CoordinateList.hh
        src/incomplete/Layers/SparseReader.cuh

        src/include/Types.cuh
        src/models/GlassEtch/include/encode.cuh
        src/include/DeviceMatrix.cuh
        src/models/GlassEtch/include/scatter.cuh
        src/tests/network/network.hh
        src/tests/network/Network.hh
        src/include/sigmoid.hh
        src/include/macros.h
        src/tests/network/FeedbackLoop.hh
        src/tests/network/NetworkLogicAND.hh
        src/tests/network/ORGate.hh
        src/tests/network/NetworkLogic.hh
        src/tests/network/Netowrl.cc
        src/tests/network/Network.inl
)

# Compile those .cc files with NVCC so standard & CUDA headers resolve
set_source_files_properties(
        src/models/GlassEtch/src/main.cc
        PROPERTIES LANGUAGE CUDA
)

target_compile_options(GlassEtch PRIVATE -O3)

target_link_libraries(GlassEtch
        PRIVATE
        $<$<BOOL:${USE_CUDA}>:${CUDAToolkit_LIBRARIES}>
        OpenMP::OpenMP_CXX
)