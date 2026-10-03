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
find_package(CUDAToolkit REQUIRED)
find_package(LAPACK REQUIRED)
if(NOT TARGET Cellerator::native_foundation OR NOT TARGET Cellerator::indexed_mechanism)
  message(FATAL_ERROR "Cellerator package lacks the native foundation or indexed mechanism target")
endif()
execute_process(COMMAND git rev-parse HEAD
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE GlassHelix_BUILD_SOURCE_REVISION
  OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _glasshelix_git_result)
string(LENGTH "${GlassHelix_BUILD_SOURCE_REVISION}" _glasshelix_revision_length)
if(NOT _glasshelix_git_result EQUAL 0 OR
   NOT _glasshelix_revision_length EQUAL 40 OR
   NOT GlassHelix_BUILD_SOURCE_REVISION MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "GlassHelix source identity requires the configured Git HEAD")
endif()
string(LENGTH "${Cellerator_BUILD_SOURCE_REVISION}" _cellerator_revision_length)
if(NOT DEFINED Cellerator_BUILD_SOURCE_REVISION OR
   NOT _cellerator_revision_length EQUAL 40 OR
   NOT Cellerator_BUILD_SOURCE_REVISION MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "Cellerator package lacks a valid build source identity")
endif()
add_library(glasshelix_core INTERFACE)
add_library(GlassHelix::core ALIAS glasshelix_core)
set_target_properties(glasshelix_core PROPERTIES EXPORT_NAME core)
target_compile_features(glasshelix_core INTERFACE cxx_std_20)
target_include_directories(glasshelix_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(glasshelix_core INTERFACE
  Cellerator::native_foundation Cellerator::indexed_mechanism)

add_library(glasshelix_foundation STATIC src/system/assembly.cc)
add_library(GlassHelix::foundation ALIAS glasshelix_foundation)
set_target_properties(glasshelix_foundation PROPERTIES EXPORT_NAME foundation)
target_compile_features(glasshelix_foundation PUBLIC cxx_std_20)
target_link_libraries(glasshelix_foundation PUBLIC GlassHelix::core)

add_library(glasshelix_experiment INTERFACE)
add_library(GlassHelix::experiment ALIAS glasshelix_experiment)
set_target_properties(glasshelix_experiment PROPERTIES EXPORT_NAME experiment)
target_link_libraries(glasshelix_experiment INTERFACE GlassHelix::foundation)

set(_glasshelix_install_targets glasshelix_core glasshelix_foundation glasshelix_experiment)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/artifacts/nf1_replay.cc" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/nf1_module.cc")
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  add_library(glasshelix_nf1_replay STATIC src/artifacts/nf1_replay.cc)
  add_library(GlassHelix::nf1_replay ALIAS glasshelix_nf1_replay)
  set_target_properties(glasshelix_nf1_replay PROPERTIES EXPORT_NAME nf1_replay)
  target_compile_features(glasshelix_nf1_replay PUBLIC cxx_std_20)
  target_compile_definitions(glasshelix_nf1_replay PRIVATE
    GLASSHELIX_NF1_SOURCE_REVISION="${GlassHelix_BUILD_SOURCE_REVISION}"
    CELLERATOR_NF1_SOURCE_REVISION="${Cellerator_BUILD_SOURCE_REVISION}")
  target_link_libraries(glasshelix_nf1_replay PUBLIC GlassHelix::core Cellerator::local_differential)
  list(APPEND _glasshelix_install_targets glasshelix_nf1_replay)

  add_executable(gh_nf1a_replay_cli tools/native_foundation/nf1_replay_cli.cc)
  target_link_libraries(gh_nf1a_replay_cli PRIVATE GlassHelix::nf1_replay)
  install(TARGETS gh_nf1a_replay_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  add_library(glasshelix_nf1 MODULE bindings/python/nf1_module.cc)
  set_target_properties(glasshelix_nf1 PROPERTIES PREFIX "")
  target_compile_features(glasshelix_nf1 PRIVATE cxx_std_20)
  target_link_libraries(glasshelix_nf1 PRIVATE Python3::Module Cellerator::local_differential)
  install(TARGETS glasshelix_nf1 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(DIRECTORY python/glasshelix DESTINATION ${CMAKE_INSTALL_LIBDIR}
          FILES_MATCHING PATTERN "*.py")
endif()

install(TARGETS ${_glasshelix_install_targets} EXPORT GlassHelixFoundationTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT GlassHelixFoundationTargets FILE GlassHelixFoundationTargets.cmake
  NAMESPACE GlassHelix:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelix)
configure_package_config_file(cmake/GlassHelixFoundationConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GlassHelixConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelix)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/GlassHelixConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GlassHelixConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/GlassHelixConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/GlassHelix)
if(BUILD_TESTING)
  add_subdirectory(tests/native_foundation/core)
  add_subdirectory(tests/native_foundation/system)
  add_subdirectory(tests/native_foundation/dynamics)
  add_subdirectory(tests/native_foundation/build)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_foundation/observation/CMakeLists.txt")
    add_subdirectory(tests/native_foundation/observation)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_foundation/inference/CMakeLists.txt")
    add_subdirectory(tests/native_foundation/inference)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_foundation/interrogation/CMakeLists.txt")
    add_subdirectory(tests/native_foundation/interrogation)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_foundation/reduction/CMakeLists.txt")
    add_subdirectory(tests/native_foundation/reduction)
  endif()
  if(TARGET glasshelix_nf1_replay)
    add_executable(gh_nf1a_replay_test tests/native_foundation/artifacts/nf1_replay_test.cc)
    target_link_libraries(gh_nf1a_replay_test PRIVATE GlassHelix::nf1_replay)
    add_test(NAME gh_nf1a_replay
      COMMAND gh_nf1a_replay_test
        "${CMAKE_CURRENT_BINARY_DIR}/nf1-replay.artifact"
        "$<TARGET_FILE:gh_nf1a_replay_cli>")
    add_test(NAME gh_nf1a_python
      COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/tests/native_foundation/python/test_nf1.py")
    set_tests_properties(gh_nf1a_python PROPERTIES
      ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:glasshelix_nf1>")
  endif()
endif()

if(BUILD_TESTING)
  enable_language(CUDA)
  add_executable(gh_nf1a_complete_program bench/native_foundation/nf1_complete_program.cu)
  target_link_libraries(gh_nf1a_complete_program PRIVATE
    GlassHelix::foundation Cellerator::native_foundation Cellerator::local_differential CUDA::cudart)
  set_target_properties(gh_nf1a_complete_program PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED YES)
  add_test(NAME gh_nf1a_complete_program COMMAND gh_nf1a_complete_program)
endif()

# Acceptance checks must execute in optimized builds as well as debug builds.
if(BUILD_TESTING)
  foreach(_nf1_test
    gh_nf1a_replay_test
    gh_nf1a_python
    gh_nf1a_complete_program
    gh_nf1_contract_quantity
    gh_nf1_contract_system
    gh_nf1_contract_request
    gh_nf1_contract_system_semantics
    gh_nf1_s01
    gh_nf1_dynamics_contract
    gh_nf1_dynamics_cuda
    gh_nf1_external_width33
    gh_nf1a_observation
    gh_nf1a_inference
    gh_nf1a_inference_cuda
    gh_nf1a_diagnostics
    gh_nf1a_alternatives
    gh_nf1a_differentiate
    gh_nf1a_reduction
  )
    if(TARGET ${_nf1_test})
      target_compile_options(${_nf1_test} PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-UNDEBUG>)
    endif()
  endforeach()
endif()
