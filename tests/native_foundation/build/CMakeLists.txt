configure_file(external_consumer.cmake.in external_consumer.cmake @ONLY)
add_test(NAME gh_nf1_external_width33
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/external_consumer.cmake)
