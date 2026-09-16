#include <GlassHelix/inference/inference.hh>

#include <cmath>
#include <stdexcept>

namespace {
using namespace glasshelix;

core::quantity observed_quantity(std::size_t extent = 1) {
  core::quantity quantity;
  quantity.axis = {{core::identity::biological_abi_version,
                    core::identity::serialized_record_kind::persistent_axis_identity,
                    sizeof(core::identity::persistent_axis_identity)},
                   {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  quantity.structure = {5, 0};
  quantity.units = "reporter";
  quantity.role = core::quantity_role::observation;
  quantity.extent = extent;
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
  const auto reporter = observation::observation_map::partial(observed, 1, {0});
  const core::observation_record evidence{observed, {0.0}, {true}, "reporter", "hours", "sample",
                                          "declared", 1.0, core::sampling_unit::population_snapshot, true};
  const observation::gaussian_noise likelihood{0.1, 1};
  inference::finite_candidates belief{{{"correlated", "m1", {0.0, 0.0}, 0.5},
                                        {"anticorrelated", "m2", {1.0, 1.0}, 0.5}}};
  belief.propagate([](const inference::candidate& value) {
    return observation::engine_output{{value.joint_state[1]}, provenance()};
  });
  belief.assimilate(evidence, likelihood, reporter, "reporter@1h");
  if (belief.values().size() != 2 || belief.values()[0].weight < .999 || belief.values()[1].weight > .001 ||
      belief.values()[0].joint_state[0] != 0.0 || belief.values()[1].joint_state[0] != 1.0 ||
      belief.values()[0].mechanism_id == belief.values()[1].mechanism_id ||
      belief.evidence_history() != std::vector<std::string>{"reporter@1h"}) return 1;

  inference::finite_candidates indistinguishable{{{"left", "m1", {2.0, 0.0}, 0.5},
                                                    {"right", "m2", {9.0, 0.0}, 0.5}}};
  indistinguishable.propagate([](const inference::candidate& value) {
    return observation::engine_output{{value.joint_state[1]}, provenance()};
  });
  indistinguishable.assimilate(evidence, likelihood, reporter, "same");
  if (std::abs(indistinguishable.values()[0].weight - .5) > 1e-12 ||
      std::abs(indistinguishable.values()[1].weight - .5) > 1e-12) return 2;

  bool zero_total = false;
  auto impossible = evidence;
  impossible.values[0] = 1e308;
  try {
    indistinguishable.assimilate(impossible, likelihood, reporter, "impossible");
  } catch (const std::domain_error&) { zero_total = true; }
  if (!zero_total) return 3;

  bool duplicate_evidence = false;
  try { indistinguishable.assimilate(evidence, likelihood, reporter, "same"); }
  catch (const std::invalid_argument&) { duplicate_evidence = true; }
  if (!duplicate_evidence) return 4;

  const auto reporters = observed_quantity(2);
  const auto paired_reporter = observation::observation_map::partial(reporters, 3, {1, 2});
  const core::observation_record paired_evidence{reporters, {0.0, 0.0}, {true, true}, "two-reporters", "hours",
                                                 "sample", "declared", 1.0,
                                                 core::sampling_unit::population_snapshot, true};
  inference::finite_candidates correlations{{{"correlated", "m-correlation", {0.0, 0.0, 0.0}, 0.5},
                                                {"anticorrelated", "m-anticorrelation", {1.0, 0.0, 1.0}, 0.5}}};
  correlations.propagate([](const inference::candidate& value) {
    return observation::engine_output{{value.joint_state[0], value.joint_state[1], value.joint_state[2]}, provenance()};
  });
  correlations.assimilate(paired_evidence, likelihood, paired_reporter, "two-reporters@1h");
  if (correlations.values().size() != 2 || correlations.values()[0].weight < .999 ||
      correlations.values()[0].hypothesis_id != "correlated" ||
      correlations.values()[1].hypothesis_id != "anticorrelated" ||
      correlations.values()[0].joint_state != std::vector<double>{0.0, 0.0, 0.0} ||
      correlations.values()[1].joint_state != std::vector<double>{1.0, 0.0, 1.0}) return 5;

  inference::finite_candidates extremes{{{"near", "m-near", {1e154}, 0.5},
                                          {"far", "m-far", {-1e154}, 0.5}}};
  extremes.propagate([](const inference::candidate& value) {
    return observation::engine_output{{value.joint_state[0]}, provenance()};
  });
  auto extreme_evidence = evidence;
  extreme_evidence.values[0] = 1e154;
  extremes.assimilate(extreme_evidence, observation::gaussian_noise{1.0, 2}, reporter, "extreme");
  if (!std::isfinite(extremes.values()[0].weight) || !std::isfinite(extremes.values()[1].weight) ||
      extremes.values()[0].weight != 1.0 || extremes.values()[1].weight != 0.0) return 6;
  return 0;
}
