#pragma once
#include <cmath>
#include <stdexcept>
namespace glasshelix::reduction { struct supplied_microstate {double left,right,reporter;}; struct reduction_result {double aggregate,derivative,reporter;bool closed;}; inline reduction_result evaluate(supplied_microstate x,double left_rate,double right_rate){if(!std::isfinite(x.left)||!std::isfinite(x.right)||!std::isfinite(x.reporter)||!std::isfinite(left_rate)||!std::isfinite(right_rate)||left_rate<0||right_rate<0)throw std::invalid_argument("invalid supplied reduction regime");return {x.left+x.right,left_rate*x.left+right_rate*x.right,x.reporter,left_rate==right_rate};} }
namespace glasshelix::reduction {
enum class threshold_result { pass, fail, unresolved };

struct error_report {
  double conversion, integration, dropped_support, empirical, certified, scientific;
  bool justified;
  const char* provenance;
  const char* source;
  const char* domain;
};

inline threshold_result assess(error_report e, double threshold) {
  if (!std::isfinite(threshold) || threshold < 0 || !e.provenance || !*e.provenance ||
      !e.source || !*e.source || !e.domain || !*e.domain)
    throw std::invalid_argument("invalid supplied error provenance");
  for (double x : {e.conversion, e.integration, e.dropped_support, e.empirical,
                   e.certified, e.scientific})
    if (!std::isfinite(x) || x < 0) throw std::invalid_argument("invalid error component");
  // Empirical and Jacobian-derived diagnostics never become a certificate.
  if (!e.justified) return threshold_result::unresolved;
  return e.certified <= threshold ? threshold_result::pass : threshold_result::fail;
}
}
