add_executable(gh_nf1_s01 s01.cc)
target_link_libraries(gh_nf1_s01 PRIVATE GlassHelix::foundation Cellerator::native_foundation)
target_compile_features(gh_nf1_s01 PRIVATE cxx_std_20)
add_test(NAME gh_nf1_s01 COMMAND gh_nf1_s01)
