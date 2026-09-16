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
find_package(Cellerator CONFIG REQUIRED COMPONENTS native_foundation)
if(NOT TARGET Cellerator::native_foundation)
  message(FATAL_ERROR "Cellerator package lacks Cellerator::native_foundation")
endif()
add_library(glasshelix_foundation STATIC src/system/assembly.cc)
add_library(GlassHelix::foundation ALIAS glasshelix_foundation)
target_compile_features(glasshelix_foundation PUBLIC cxx_std_20)
target_include_directories(glasshelix_foundation PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(glasshelix_foundation PUBLIC Cellerator::native_foundation)
install(TARGETS glasshelix_foundation EXPORT GlassHelixFoundationTargets)
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
  set(NF1_CELLERATOR_INCLUDE_DIR "" CACHE PATH "Cellerator public include directory")
  if(NOT NF1_CELLERATOR_INCLUDE_DIR)
    get_target_property(NF1_CELLERATOR_INCLUDE_DIR Cellerator::native_foundation INTERFACE_INCLUDE_DIRECTORIES)
  endif()
  add_subdirectory(tests/native_foundation/core)
  add_subdirectory(tests/native_foundation/system)
endif()
