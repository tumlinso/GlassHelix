#include <GlassHelix/analysis/requests.hh>
#include <array>
#include <iostream>

using namespace glasshelix;
void require(bool value) {
  if (!value) throw std::runtime_error("analysis contract check failed");
}
template<class F> void rejected(F action) {
  bool caught = false;
  try { action(); } catch (const std::invalid_argument&) { caught = true; }
  require(caught);
}

int main() {
  namespace df = cellerator::compute::differential;
  std::array<double, 2> left{2, 3}, right{5, 7}, dl{1, 2}, dr{4, 6},
                        cotangent{1, 2}, output{}, left_adjoint{}, right_adjoint{};
  df::local_binding<double> binding{left, right, dl, dr, cotangent, output, left_adjoint, right_adjoint};
  analysis::response_context context{{core::differentiated_object::vector_field,
      core::forward_request{}, {}}, "whole-hypothesis", "response-evidence", "CE-local-differential",
      {"state", "direction", "declared", analysis::perturbation_role::state}};
  interrogation::vector_field_primal saved{1, 2, 3, 4, interrogation::primal_policy::saved};
  auto result = analysis::local_response(context, saved, saved, df::numeric::local_operation::multiply, binding);
  require(result.status == analysis::assessment::qualified && output[0] == 13 && output[1] == 32);
  result = analysis::local_response(context, saved, saved, df::numeric::local_operation::multiply, binding, true);
  require(result.status == analysis::assessment::qualified && left_adjoint[0] == 5 &&
          left_adjoint[1] == 14 && right_adjoint[0] == 2 && right_adjoint[1] == 6);
  auto live = saved;
  ++live.parameters;
  rejected([&] { analysis::local_response(context, saved, live, df::numeric::local_operation::multiply, binding); });
  context.request.object = core::differentiated_object::implemented_rollout;
  require(analysis::local_response(context, saved, saved, df::numeric::local_operation::multiply, binding).status
          == analysis::assessment::unsupported);
  interrogation::supplied_alternative a{"a", "whole-a", "prediction", {"reporter"}, {2}, {}};
  interrogation::supplied_alternative b{"b", "whole-b", "prediction", {"reporter"}, {2}, {}};
  const auto comparison = analysis::compare_supplied(a, b, 1, 10, "comparison-evidence");
  require(comparison.local.locally_degenerate && !comparison.local.globally_identified);
  b.selected_edges = {"mixed-edge"};
  rejected([&] { analysis::compare_supplied(a, b, 1, 10, "comparison-evidence"); });
  std::cout << "real CE host JVP/VJP, generation admission and atomic comparison passed\n";
}
