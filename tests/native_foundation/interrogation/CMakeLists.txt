add_executable(gh_nf1a_diagnostics diagnostics.cc)
target_link_libraries(gh_nf1a_diagnostics PRIVATE GlassHelix::core LAPACK::LAPACK)
add_test(NAME gh_nf1a_diagnostics COMMAND gh_nf1a_diagnostics)
