#include <GlassHelix/inference/inference.hh>

#include <cmath>
#include <stdexcept>

namespace {
using namespace glasshelix;

core::quantity observed_quantity() {
  core::quantity quantity;
  quantity.axis = {{core::identity::biological_abi_version,
                    core::identity::serialized_record_kind::persistent_axis_identity,
                    sizeof(core::identity::persistent_axis_identity)},
                   {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  quantity.structure = {5, 0};
  quantity.units = "reporter";
  quantity.role = core::quantity_role::observation;
  quantity.extent = 1;
  return quantity;
}

core::result_provenance provenance() {
  core::result_provenance result;
  result.scientific_model_id = "system";
  result.model_revision = "r1";
  result.gh_source_commit = "gh";
  result.ce_source_commit = "ce";
  result.numerical_policy = "fp32";
  return result;
}
}  // namespace

int main() {
  using namespace glasshelix;
  const auto observed = observed_quantity();
  const core::observation_record evidence{observed, {0.0}, {true}, "reporter", "hours", "sample",
                                          "declared", 1.0, core::sampling_unit::population_snapshot, true};
  const observation::gaussian_noise likelihood{0.1, 1};
  inference::finite_candidates belief{{{"correlated", "m1", {0.0, 0.0}, 0.5},
                                        {"anticorrelated", "m2", {1.0, 1.0}, 0.5}}};
  belief.propagate([](const inference::candidate& value) {
    return observation::engine_output{{value.joint_state[1]}, provenance()};
  });
  belief.assimilate(evidence, likelihood,
                    [](const observation::engine_output& value) { return value.values; }, "reporter@1h");
  if (belief.values().size() != 2 || belief.values()[0].weight < .999 || belief.values()[1].weight > .001 ||
      belief.values()[0].joint_state[0] != 0.0 || belief.values()[1].joint_state[0] != 1.0 ||
      belief.values()[0].mechanism_id == belief.values()[1].mechanism_id ||
      belief.evidence_history() != std::vector<std::string>{"reporter@1h"}) return 1;

  inference::finite_candidates indistinguishable{{{"left", "m1", {2.0, 0.0}, 0.5},
                                                    {"right", "m2", {9.0, 0.0}, 0.5}}};
  indistinguishable.propagate([](const inference::candidate& value) {
    return observation::engine_output{{value.joint_state[1]}, provenance()};
  });
  indistinguishable.assimilate(evidence, likelihood,
                               [](const observation::engine_output& value) { return value.values; }, "same");
  if (std::abs(indistinguishable.values()[0].weight - .5) > 1e-12 ||
      std::abs(indistinguishable.values()[1].weight - .5) > 1e-12) return 2;

  bool zero_total = false;
  auto impossible = evidence;
  impossible.values[0] = 1e308;
  try {
    indistinguishable.assimilate(impossible, likelihood,
                                 [](const observation::engine_output& value) { return value.values; }, "impossible");
  } catch (const std::domain_error&) { zero_total = true; }
  if (!zero_total) return 3;
  return 0;
}
