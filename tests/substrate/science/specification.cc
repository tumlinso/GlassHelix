#include <GlassHelix/experiment/specification.hh>
#include <iostream>

using namespace glasshelix;

void require(bool condition) {
  if (!condition) throw std::runtime_error("scientific specification check failed");
}
template<class F> void rejected(F action) {
  bool caught = false;
  try { action(); } catch (const std::invalid_argument&) { caught = true; }
  require(caught);
}
core::quantity quantity(core::quantity_role role, std::size_t extent) {
  core::quantity q;
  q.axis = {{core::identity::biological_abi_version,
             core::identity::serialized_record_kind::persistent_axis_identity,
             sizeof(core::identity::persistent_axis_identity)}, {1, 0}, {2, 0}, {3, 0}, {4, 0}};
  q.structure = {5, 0};
  q.units = "declared";
  q.role = role;
  q.extent = extent;
  return q;
}

int main() {
  const auto state = quantity(core::quantity_role::dynamical_state, 2);
  const auto parameter = quantity(core::quantity_role::parameter, 1);
  const auto observed = quantity(core::quantity_role::observation, 2);
  auto system = std::make_shared<const core::system_definition>(
      "scientific-model", std::vector<core::quantity>{state, parameter, observed},
      std::vector<core::mechanism>{{"m1", {0, 1}, {"state", "parameter"}, {2}},
                                    {"m2", {0, 1}, {"state", "parameter"}, {2}}});
  models::candidate_set hypotheses({{"h1", "m1", {1, 2}, .5}, {"h2", "m2", {8, 9}, .5}});
  experiment::specification spec(system, hypotheses);
  require(spec.system().quantities()[0].role != spec.system().quantities()[1].role);
  experiment::measurement measurement{
      {observed, {0, std::numeric_limits<double>::quiet_NaN()}, {true, false},
       "RNA", "hours", "sample", "snapshot", 1,
       core::sampling_unit::population_snapshot, true}, {false, true}, "RNA@1"};
  spec.validate_measurement(measurement);
  auto invalid = measurement;
  invalid.observation.present[1] = true;
  rejected([&] { spec.validate_measurement(invalid); });
  invalid = measurement;
  invalid.detection_support.clear();
  rejected([&] { spec.validate_measurement(invalid); });
  require(!core::admissible_individual_successor(measurement.observation, measurement.observation));
  experiment::record receipt;
  receipt.experiment_id = "fixture";
  receipt.source = {data::source_kind::synthetic_fixture, "fixture", "audit", {"inputs"}};
  receipt.provenance.scientific_model_id = system->id();
  receipt.provenance.model_revision = "1";
  receipt.provenance.gh_source_commit = "host-test";
  receipt.provenance.ce_source_commit = "published-ce-header";
  receipt.provenance.numerical_policy = "host scientific contract";
  // Reuse the actual finite-candidate conditioning owner. The declared local
  // outputs agree on the measured channel; the unmeasured channel cannot select
  // a mechanism or mix the incompatible hidden-state coordinates.
  hypotheses.propagate([&](const auto& h) {
    return observation::engine_output{{0, h.joint_state[0]}, receipt.provenance};
  });
  const auto map = observation::observation_map::partial(observed, 2, {0, 1});
  hypotheses.assimilate(measurement.observation, {1, 1}, map, measurement.evidence_id);
  require(hypotheses.values()[0].weight == .5 && hypotheses.values()[1].weight == .5);
  std::vector<experiment::prediction> predictions{
      {"h1", observed, {{0, 1}, receipt.provenance}},
      {"h2", observed, {{0, 8}, receipt.provenance}}};
  const auto result = spec.snapshot(receipt, hypotheses, predictions);
  require(result.whole_hypotheses[0].joint_state == std::vector<double>({1, 2}));
  require(result.whole_hypotheses[1].joint_state == std::vector<double>({8, 9}));
  require(result.conditioning_evidence_ids == std::vector<std::string>{"RNA@1"});
  rejected([&] { spec.snapshot(receipt, hypotheses, predictions,
                              experiment::interpretation_scope::globally_identified_mechanism); });
  predictions[1].hypothesis_id = "h1";
  rejected([&] { spec.snapshot(receipt, hypotheses, predictions); });
  models::candidate_set mixed({{"h1", "m2", {1, 2}, .5}, {"h2", "m2", {8, 9}, .5}});
  rejected([&] { spec.snapshot(receipt, mixed, {}); });
  models::candidate_set mixed_state({{"h1", "m1", {1, 9}, .5}, {"h2", "m2", {8, 9}, .5}});
  rejected([&] { spec.snapshot(receipt, mixed_state, {}); });
  models::candidate_set duplicate({{"h1", "m1", {1, 2}, .5}, {"h1", "m2", {8, 9}, .5}});
  rejected([&] { experiment::specification bad(system, duplicate); });
  std::cout << "scientific roles, masked conditioning, whole alternatives and scope checks passed\n";
}
