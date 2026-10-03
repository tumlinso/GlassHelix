#pragma once

#include <GlassHelix/analysis/result.hh>
#include <GlassHelix/interrogation/alternatives.hh>
#include <GlassHelix/interrogation/differentiation.hh>

namespace glasshelix::analysis {

enum class perturbation_role { state, parameter, forcing, activity };
struct perturbation {
  std::string quantity_id, direction_id, units;
  perturbation_role role = perturbation_role::state;
};
struct response_context {
  core::response_request request;
  std::string whole_hypothesis_id, evidence_id, provider_evidence_id;
  perturbation supplied_direction;

  void validate() const {
    if (whole_hypothesis_id.empty() || evidence_id.empty() || provider_evidence_id.empty() ||
        supplied_direction.quantity_id.empty() || supplied_direction.direction_id.empty() ||
        supplied_direction.units.empty())
      throw std::invalid_argument("response requires whole-hypothesis, perturbation and provider provenance");
  }
};

// A bounded host local derivative action; CE owns formulas and borrowed spans.
// No solver, derivative graph, buffer, stream or numerical state is owned here.
template<class T>
capability_result local_response(
    const response_context& context, interrogation::vector_field_primal saved,
    interrogation::vector_field_primal live,
    cellerator::compute::differential::numeric::local_operation operation,
    const cellerator::compute::differential::local_binding<T>& binding,
    bool reverse = false) {
  context.validate();
  const std::string action = reverse ? "local_vjp" : "local_jvp";
  if (context.request.object != core::differentiated_object::vector_field ||
      !std::holds_alternative<core::forward_request>(context.request.evolution) ||
      !context.request.selected_outputs.empty())
    return {action, assessment::unsupported, context.evidence_id};
  interrogation::require_current(saved, live);
  namespace df = cellerator::compute::differential;
  const auto status = reverse ? df::local_vjp(operation, binding) : df::local_jvp(operation, binding);
  if (status != df::numeric::local_status::success)
    throw std::runtime_error("Cellerator rejected the local response binding");
  return {action, assessment::qualified, context.evidence_id};
}

struct comparison_receipt {
  interrogation::alternative_comparison local;
  std::string evidence_id, uncertainty_scope;
};

inline comparison_receipt compare_supplied(
    const interrogation::supplied_alternative& first,
    const interrogation::supplied_alternative& second,
    std::size_t first_runs, std::size_t second_runs, std::string evidence_id) {
  if (evidence_id.empty()) throw std::invalid_argument("comparison evidence identity required");
  return {interrogation::compare_atomic_alternatives(first, second, first_runs, second_runs),
          std::move(evidence_id), "local supplied outputs; global identification unresolved"};
}

}  // namespace glasshelix::analysis
