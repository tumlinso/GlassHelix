#include <GlassHelix/inference/inference.hh>

#include <cmath>
#include <limits>
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

  // A finite log-likelihood gap of 800 underflows the display weight. The
  // independent opposite observation must still restore equal support.
  const auto make_recoverable = [] {
    inference::finite_candidates result{{{"left", "m-left", {0.0}, 0.5},
                                           {"right", "m-right", {40.0}, 0.5}}};
    result.propagate([](const inference::candidate& value) {
      return observation::engine_output{value.joint_state, provenance()};
    });
    return result;
  };
  const observation::gaussian_noise unit_noise{1.0, 3};
  auto opposite = evidence;
  opposite.values[0] = 40.0;
  auto recoverable = make_recoverable();
  recoverable.assimilate(evidence, unit_noise, reporter, "left");
  if (recoverable.values()[0].weight != 1.0 || recoverable.values()[1].weight != 0.0) return 7;
  const auto before_failure = recoverable.values();
  const auto history_before_failure = recoverable.evidence_history();
  bool failed_update = false;
  try { recoverable.assimilate(impossible, unit_noise, reporter, "failed"); }
  catch (const std::domain_error&) { failed_update = true; }
  if (!failed_update || recoverable.evidence_history() != history_before_failure ||
      recoverable.values()[0].weight != before_failure[0].weight ||
      recoverable.values()[1].weight != before_failure[1].weight) return 8;
  // Reuse the failed ID to prove that failure did not record evidence. Recovery
  // also proves that the internal log weights survived the failed update.
  recoverable.assimilate(opposite, unit_noise, reporter, "failed");
  if (std::abs(recoverable.values()[0].weight - .5) > 1e-12 ||
      std::abs(recoverable.values()[1].weight - .5) > 1e-12) return 9;

  auto reversed = make_recoverable();
  reversed.assimilate(opposite, unit_noise, reporter, "right");
  reversed.assimilate(evidence, unit_noise, reporter, "left");
  const auto two_coordinates = observation::observation_map::partial(reporters, 2, {0, 1});
  auto batch_evidence = paired_evidence;
  batch_evidence.values = {0.0, 40.0};
  inference::finite_candidates batched{{{"left", "m-left", {0.0, 0.0}, 0.5},
                                         {"right", "m-right", {40.0, 40.0}, 0.5}}};
  batched.propagate([](const inference::candidate& value) {
    return observation::engine_output{value.joint_state, provenance()};
  });
  batched.assimilate(batch_evidence, unit_noise, two_coordinates, "batch");
  for (std::size_t index = 0; index != 2; ++index) {
    if (std::abs(reversed.values()[index].weight - recoverable.values()[index].weight) > 1e-12 ||
        std::abs(batched.values()[index].weight - recoverable.values()[index].weight) > 1e-12) return 10;
  }
  recoverable.assimilate(opposite, unit_noise, reporter, "right-again");
  if (recoverable.values()[1].weight != 1.0 || recoverable.values()[0].weight != 0.0) return 11;

  inference::finite_candidates zero_prior{{{"included", "m-left", {0.0}, 1.0},
                                            {"excluded", "m-right", {40.0}, 0.0}}};
  zero_prior.propagate([](const inference::candidate& value) {
    return observation::engine_output{value.joint_state, provenance()};
  });
  zero_prior.assimilate(opposite, unit_noise, reporter, "excluded-favored");
  if (zero_prior.values()[0].weight != 1.0 || zero_prior.values()[1].weight != 0.0) return 12;

  // Positive supplied priors also survive underflow during initial display
  // normalization, before any evidence has been assimilated.
  const auto tiny_prior = std::numeric_limits<double>::denorm_min();
  const auto compensating_mean = std::sqrt(2.0 * (std::log(1e308) - std::log(tiny_prior)));
  inference::finite_candidates tiny{{{"tiny", "m-tiny", {compensating_mean}, tiny_prior},
                                      {"large", "m-large", {0.0}, 1e308}}};
  if (tiny.values()[0].weight != 0.0 || tiny.values()[1].weight != 1.0) return 13;
  tiny.propagate([](const inference::candidate& value) {
    return observation::engine_output{value.joint_state, provenance()};
  });
  auto compensating_evidence = evidence;
  compensating_evidence.values[0] = compensating_mean;
  tiny.assimilate(compensating_evidence, unit_noise, reporter, "compensating");
  if (std::abs(tiny.values()[0].weight - .5) > 1e-10 ||
      std::abs(tiny.values()[1].weight - .5) > 1e-10) return 14;
  return 0;
}
