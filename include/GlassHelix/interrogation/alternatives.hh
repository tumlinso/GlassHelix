#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace glasshelix::interrogation {

// A supplied candidate is atomic: it names one complete branch. Selecting
// individual branch edges is deliberately outside this local comparison API.
struct supplied_alternative {
  std::string candidate_id;
  std::string whole_branch_id;
  std::string role;
  std::vector<std::string> shared_features;
  std::vector<double> local_output;
  std::vector<std::string> selected_edges;
};

struct alternative_comparison {
  std::string first_candidate_id;
  std::string second_candidate_id;
  bool locally_degenerate;
  bool globally_identified;
  std::size_t first_optimization_runs;
  std::size_t second_optimization_runs;
};

inline alternative_comparison compare_atomic_alternatives(
    const supplied_alternative& first, const supplied_alternative& second,
    std::size_t first_optimization_runs, std::size_t second_optimization_runs) {
  if (first.candidate_id.empty() || second.candidate_id.empty() ||
      first.candidate_id == second.candidate_id || first.whole_branch_id.empty() ||
      second.whole_branch_id.empty() || first.whole_branch_id == second.whole_branch_id ||
      first.role.empty() || first.role != second.role ||
      first.shared_features.empty() || first.shared_features != second.shared_features ||
      !first.selected_edges.empty() || !second.selected_edges.empty() ||
      first.local_output.size() != second.local_output.size())
    throw std::invalid_argument("unaligned or non-atomic supplied alternatives");

  bool locally_degenerate = true;
  for (std::size_t i = 0; i < first.local_output.size(); ++i) {
    if (!std::isfinite(first.local_output[i]) || !std::isfinite(second.local_output[i]))
      throw std::invalid_argument("non-finite local alternative output");
    locally_degenerate &= first.local_output[i] == second.local_output[i];
  }
  // Optimization runs are diagnostic only. They never encode probability or
  // establish global identification from a local comparison.
  return {first.candidate_id, second.candidate_id, locally_degenerate, false,
          first_optimization_runs, second_optimization_runs};
}

}  // namespace glasshelix::interrogation
