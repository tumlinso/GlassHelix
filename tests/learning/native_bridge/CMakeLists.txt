cmake_minimum_required(VERSION 3.28)
project(GHML2NativeBridge LANGUAGES CXX)
find_package(GlassHelix CONFIG REQUIRED)
find_package(Cellerator CONFIG REQUIRED COMPONENTS native_foundation indexed_mechanism local_differential)
add_executable(gh_ml2_native_bridge bridge.cc)
target_compile_features(gh_ml2_native_bridge PRIVATE cxx_std_20)
target_link_libraries(gh_ml2_native_bridge PRIVATE GlassHelix::core Cellerator::local_differential)
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sdk-archives.txt" CONTENT
"$<TARGET_FILE:Cellerator::native_numeric>\n$<TARGET_FILE:Cellerator::local_differential>\n$<TARGET_FILE:Cellerator::indexed_mechanism>\n$<TARGET_FILE:Cellerator::executable_program>\n")
if(NOT GH_BRIDGE_SOURCE_COMMIT MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "GH_BRIDGE_SOURCE_COMMIT required")
endif()
target_compile_definitions(gh_ml2_native_bridge PRIVATE
  GH_SOURCE_COMMIT="${GH_BRIDGE_SOURCE_COMMIT}"
  CE_SOURCE_COMMIT="${Cellerator_BUILD_SOURCE_REVISION}")
