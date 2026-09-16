cmake_minimum_required(VERSION 3.28)
project(GlassHelix VERSION 0.1 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# GlassHelix consumes a published Cellerator library. It never compiles
# Cellerator sources or provides an accelerator substitute.
find_package(Cellerator CONFIG REQUIRED COMPONENTS native_foundation indexed_mechanism)
if(NOT TARGET Cellerator::native_foundation OR NOT TARGET Cellerator::indexed_mechanism)
  message(FATAL_ERROR "Cellerator package lacks the native foundation or indexed mechanism target")
endif()
add_library(glasshelix_core INTERFACE)
add_library(GlassHelix::core ALIAS glasshelix_core)
target_compile_features(glasshelix_core INTERFACE cxx_std_20)
target_include_directories(glasshelix_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(glasshelix_core INTERFACE
  Cellerator::native_foundation Cellerator::indexed_mechanism)

add_library(glasshelix_foundation STATIC src/system/assembly.cc)
add_library(GlassHelix::foundation ALIAS glasshelix_foundation)
target_compile_features(glasshelix_foundation PUBLIC cxx_std_20)
target_link_libraries(glasshelix_foundation PUBLIC GlassHelix::core)
install(TARGETS glasshelix_core glasshelix_foundation EXPORT GlassHelixFoundationTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT GlassHelixFoundationTargets FILE GlassHelixFoundationTargets.cmake
  NAMESPACE GlassHelix:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelix)
configure_package_config_file(cmake/GlassHelixFoundationConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GlassHelixFoundationConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelix)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/GlassHelixFoundationConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GlassHelixFoundationConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/GlassHelixFoundationConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelix)
if(BUILD_TESTING)
  add_subdirectory(tests/native_foundation/core)
  add_subdirectory(tests/native_foundation/system)
  add_subdirectory(tests/native_foundation/dynamics)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_foundation/observation/CMakeLists.txt")
    add_subdirectory(tests/native_foundation/observation)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_foundation/inference/CMakeLists.txt")
    add_subdirectory(tests/native_foundation/inference)
  endif()
endif()
