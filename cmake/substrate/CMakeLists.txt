cmake_minimum_required(VERSION 3.28)
project(GlassHelixSubstrate VERSION 0.1 LANGUAGES CXX)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

get_filename_component(GLASSHELIX_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
option(GLASSHELIX_SUBSTRATE_INSTALL_FRAMEWORK "Install optional framework clients and CLI" ON)
option(GLASSHELIX_SUBSTRATE_BUILD_NATIVE "Build assembly against a qualified installed CE package" OFF)
option(GLASSHELIX_SUBSTRATE_BUILD_ANALYSIS "Expose the qualified CE local differential component" OFF)
option(GLASSHELIX_REQUIRE_INTEGRATED "Require the accepted scientific/data/model/analysis source merge" OFF)
if(GLASSHELIX_REQUIRE_INTEGRATED)
  if(NOT GLASSHELIX_SUBSTRATE_INSTALL_FRAMEWORK)
    message(FATAL_ERROR "Integrated qualification requires the framework component")
  endif()
  foreach(_required
      experiment/specification.py data/cohort.py models/client.py
      analysis/requests.py refactoring/proposal.py)
    if(NOT EXISTS "${GLASSHELIX_SOURCE_ROOT}/python/glasshelix/${_required}")
      message(FATAL_ERROR "Accepted integrated source is missing: ${_required}")
    endif()
  endforeach()
endif()

add_library(glasshelix_substrate_contracts INTERFACE)
set_target_properties(glasshelix_substrate_contracts PROPERTIES EXPORT_NAME contracts)
target_compile_features(glasshelix_substrate_contracts INTERFACE cxx_std_20)
target_include_directories(glasshelix_substrate_contracts INTERFACE
  $<BUILD_INTERFACE:${GLASSHELIX_SOURCE_ROOT}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
install(TARGETS glasshelix_substrate_contracts EXPORT GlassHelixSubstrateTargets)
install(DIRECTORY "${GLASSHELIX_SOURCE_ROOT}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(_gh_native_json false)
if(GLASSHELIX_SUBSTRATE_BUILD_NATIVE)
  find_package(Cellerator CONFIG REQUIRED COMPONENTS native_foundation indexed_mechanism)
  if(NOT TARGET Cellerator::native_foundation OR NOT TARGET Cellerator::indexed_mechanism)
    message(FATAL_ERROR "Installed CE package lacks required native targets")
  endif()
  add_library(glasshelix_substrate_native STATIC "${GLASSHELIX_SOURCE_ROOT}/src/system/assembly.cc")
  set_target_properties(glasshelix_substrate_native PROPERTIES EXPORT_NAME native)
  target_link_libraries(glasshelix_substrate_native PUBLIC glasshelix_substrate_contracts
    Cellerator::native_foundation Cellerator::indexed_mechanism)
  install(TARGETS glasshelix_substrate_native EXPORT GlassHelixSubstrateNativeTargets)
  install(EXPORT GlassHelixSubstrateNativeTargets FILE GlassHelixSubstrateNativeTargets.cmake
    NAMESPACE GlassHelixSubstrate:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelixSubstrate")
  set(_gh_native_json true)
endif()

if(GLASSHELIX_SUBSTRATE_BUILD_ANALYSIS)
  find_package(Cellerator CONFIG REQUIRED COMPONENTS local_differential)
  find_package(CUDAToolkit REQUIRED)
  if(NOT TARGET Cellerator::local_differential)
    message(FATAL_ERROR "Installed CE package lacks its local differential target")
  endif()
  add_library(glasshelix_substrate_analysis INTERFACE)
  set_target_properties(glasshelix_substrate_analysis PROPERTIES EXPORT_NAME analysis)
  target_link_libraries(glasshelix_substrate_analysis INTERFACE glasshelix_substrate_contracts
    Cellerator::local_differential CUDA::cudart)
  install(TARGETS glasshelix_substrate_analysis EXPORT GlassHelixSubstrateAnalysisTargets)
  install(EXPORT GlassHelixSubstrateAnalysisTargets FILE GlassHelixSubstrateAnalysisTargets.cmake
    NAMESPACE GlassHelixSubstrate:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelixSubstrate")
endif()

if(GLASSHELIX_SUBSTRATE_INSTALL_FRAMEWORK)
  install(DIRECTORY "${GLASSHELIX_SOURCE_ROOT}/python/glasshelix"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/python" FILES_MATCHING PATTERN "*.py")
  configure_file(glasshelix-substrate.in glasshelix-substrate @ONLY)
  install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/glasshelix-substrate" DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${GLASSHELIX_SOURCE_ROOT}"
  OUTPUT_VARIABLE _gh_source_revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
file(GLOB_RECURSE _gh_python_sources "${GLASSHELIX_SOURCE_ROOT}/python/glasshelix/*.py")
set(_gh_hashes "")
foreach(_source IN LISTS _gh_python_sources)
  file(RELATIVE_PATH _relative "${GLASSHELIX_SOURCE_ROOT}" "${_source}")
  file(SHA256 "${_source}" _sha)
  if(_gh_hashes)
    string(APPEND _gh_hashes ",")
  endif()
  string(APPEND _gh_hashes "\"${_relative}\":\"${_sha}\"")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/install-manifest.json"
  "{\"schema\":1,\"source_revision\":\"${_gh_source_revision}\",\"native_available\":${_gh_native_json},\"python_source_sha256\":{${_gh_hashes}}}\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/install-manifest.json"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/glasshelix-substrate")

install(EXPORT GlassHelixSubstrateTargets FILE GlassHelixSubstrateTargets.cmake
  NAMESPACE GlassHelixSubstrate:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelixSubstrate")
configure_package_config_file(GlassHelixSubstrateConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/GlassHelixSubstrateConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelixSubstrate")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/GlassHelixSubstrateConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GlassHelixSubstrateConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/GlassHelixSubstrateConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelixSubstrate")
