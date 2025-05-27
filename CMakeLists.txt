cmake_minimum_required(VERSION 3.29)

# ─────────────────────────────────────────────────────────────────────────────
# Project-wide configuration
# ─────────────────────────────────────────────────────────────────────────────
project(GlassHelix LANGUAGES CXX CUDA)

set(CUDA_DIR "/usr/local/cuda")
option(USE_CUDA "Enable CUDA support" ON)

# Export compile_commands.json for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use clang++ and C++23
set(CMAKE_CXX_COMPILER       "clang++")
set(CMAKE_CXX_STANDARD       23)
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
  enable_language(CUDA)
  set(CMAKE_CUDA_COMPILER       "${CUDA_DIR}/bin/nvcc")
  set(CMAKE_CUDA_HOST_COMPILER  "${CMAKE_CXX_COMPILER}")
  set(CMAKE_CUDA_ARCHITECTURES  86)   # Ampere/Hopper
  add_compile_options(
          $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
          $<$<COMPILE_LANGUAGE:CUDA>:-stdlib=libc++>
  )

  set(CMAKE_CUDA_STANDARD       20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS     OFF)

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
        src/memory/SparseMatrix/CsrDevice.hh

        src/DeviceMacros.hh
        src/TranscriptomeEncoder.hh

        src/models/GlassEtch/src/GlassEtch.hh
        src/models/GlassEtch/src/main.cc
)

# Any TU including *.cuh must be compiled with NVCC:
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
        src/models/GlassEtch/src/GlassEtch.hh
        src/models/GlassEtch/src/main.cc

        src/memory/AsynchronousArray.hh
        src/memory/SparseMatrix/CompressedSparseRow.hh
        src/memory/SparseMatrix/IRandomAccess.hh
        src/memory/SparseMatrix/CoordinateList.hh
        src/incomplete/Layers/SparseReader.cuh
        src/memory/SparseMatrix/CsrDevice.hh

        src/models/GlassEtch/include/types.cuh
)

# Compile those .cc files with NVCC so they see both <cstdint> and CUDA headers
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

