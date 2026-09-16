#include <GlassHelix/observation/observation.hh>

#include <cmath>
#include <array>
#include <limits>
#include <stdexcept>

namespace {
using namespace glasshelix;

core::quantity observed_quantity(std::size_t extent) {
  core::quantity quantity;
  quantity.axis = {{core::identity::biological_abi_version,
                    core::identity::serialized_record_kind::persistent_axis_identity,
                    sizeof(core::identity::persistent_axis_identity)},
                   {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  quantity.structure = {5, 0};
  quantity.units = "arbitrary";
  quantity.role = core::quantity_role::observation;
  quantity.extent = extent;
  return quantity;
}

core::result_provenance provenance() {
  core::result_provenance result;
  result.scientific_model_id = "supplied-system";
  result.model_revision = "r1";
  result.gh_source_commit = "gh-source";
  result.ce_source_commit = "ce-source";
  result.numerical_policy = "fp32 baseline";
  return result;
}
}  // namespace

int main() {
  using namespace glasshelix;
  const auto output = observed_quantity(2);
  const observation::engine_output engine{{2.0, 3.0, 5.0}, provenance()};
  const auto partial = observation::observation_map::partial(output, 3, {2, 0});
  const auto selected = partial.evaluate(engine);
  if (selected != std::vector<double>{5.0, 2.0}) return 1;

  const auto linear = observation::observation_map::linear(output, 3, {1, 0, 1, 0, 2, -1});
  const auto mixed = linear.evaluate(engine);
  if (std::abs(mixed[0] - 7.0) > 1e-12 || std::abs(mixed[1] - 1.0) > 1e-12) return 2;

  const auto nonlinear = observation::observation_map::nonlinear(
      output, 3, [](std::span<const double> values) { return std::vector<double>{values[0] * values[1], values[2]}; });
  if (nonlinear.derivatives() != observation::derivative_capability::unavailable ||
      nonlinear.evaluate(engine) != std::vector<double>{6.0, 5.0}) return 3;

  core::observation_record record{output, {0.0, 41.0}, {true, false}, "reporter", "hours",
                                  "cell-1", "declared Gaussian", 2.0,
                                  core::sampling_unit::individual, false};
  if (!record.valid() || record.values[0] != 0.0 || record.present[1]) return 4;
  const observation::gaussian_noise noise{0.5, 42};
  const std::array<double, 2> likelihood_mean{0.0, 999.0};
  if (std::abs(noise.log_likelihood(record, likelihood_mean) -
               (-std::log(0.5) - 0.5 * 1.8378770664093454836)) > 1e-12) return 5;
  const std::array<double, 2> sampling_mean{1.0, 999.0};
  const auto sample_a = noise.sample(record, sampling_mean);
  const auto sample_b = noise.sample(record, sampling_mean);
  if (sample_a.values != sample_b.values || sample_a.values[1] != 41.0) return 6;

  bool rejected_nonfinite_evidence = false;
  auto nonfinite = record;
  nonfinite.values[0] = std::numeric_limits<double>::infinity();
  try { (void)noise.log_likelihood(nonfinite, likelihood_mean); }
  catch (const std::invalid_argument&) { rejected_nonfinite_evidence = true; }
  if (!rejected_nonfinite_evidence) return 7;

  bool rejected_population_history = false;
  auto population = record;
  population.unit = core::sampling_unit::population_snapshot;
  auto later = population;
  later.time = 3.0;
  rejected_population_history = !core::admissible_individual_successor(population, later);
  if (!rejected_population_history) return 8;
  return 0;
}
