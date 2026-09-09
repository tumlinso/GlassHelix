add_executable(gh_nf1_b01 b01.cc)
target_link_libraries(gh_nf1_b01 PRIVATE GlassHelix::foundation)
add_test(NAME gh_nf1_b01 COMMAND gh_nf1_b01)
