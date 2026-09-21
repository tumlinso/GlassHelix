add_executable(gh_nf1a_reduction contracts.cc)
target_link_libraries(gh_nf1a_reduction PRIVATE GlassHelix::core)
add_test(NAME gh_nf1a_reduction COMMAND gh_nf1a_reduction)
