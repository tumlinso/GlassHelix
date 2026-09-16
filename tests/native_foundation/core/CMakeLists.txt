if(NOT IS_DIRECTORY "${NF1_CELLERATOR_INCLUDE_DIR}/Cellerator")
  message(FATAL_ERROR "NF1_CELLERATOR_INCLUDE_DIR must identify Cellerator public headers")
endif()

foreach(case IN ITEMS quantity system request system_semantics)
  add_executable(gh_nf1_contract_${case} ${case}.cc)
  target_compile_features(gh_nf1_contract_${case} PRIVATE cxx_std_20)
  target_include_directories(gh_nf1_contract_${case} PRIVATE
    "${PROJECT_SOURCE_DIR}/include" "${NF1_CELLERATOR_INCLUDE_DIR}")
  add_test(NAME gh_nf1_contract_${case} COMMAND gh_nf1_contract_${case})
endforeach()
