#include <GlassHelix/interrogation/alternatives.hh>

int main() {
  using namespace glasshelix::interrogation;
  supplied_alternative a{"candidate-a", "branch-a", "reporter", {"state", "dose"}, {2., 3.}, {}};
  supplied_alternative b{"candidate-b", "branch-b", "reporter", {"state", "dose"}, {2., 3.}, {}};
  const auto equal = compare_atomic_alternatives(a, b, 1, 97);
  if (!equal.locally_degenerate || equal.globally_identified ||
      equal.first_candidate_id != "candidate-a" || equal.second_candidate_id != "candidate-b" ||
      equal.first_optimization_runs != 1 || equal.second_optimization_runs != 97) return 1;

  bool role_mismatch = false, feature_mismatch = false, hybrid = false;
  auto wrong_role = b;
  wrong_role.role = "mechanism";
  try { (void)compare_atomic_alternatives(a, wrong_role, 1, 1); } catch (...) { role_mismatch = true; }
  auto wrong_feature = b;
  wrong_feature.shared_features = {"state", "time"};
  try { (void)compare_atomic_alternatives(a, wrong_feature, 1, 1); } catch (...) { feature_mismatch = true; }
  auto edge_hybrid = b;
  edge_hybrid.selected_edges = {"branch-a-edge", "branch-b-edge"};
  try { (void)compare_atomic_alternatives(a, edge_hybrid, 1, 1); } catch (...) { hybrid = true; }
  return role_mismatch && feature_mismatch && hybrid ? 0 : 2;
}
