#include <GlassHelix/experiment/record.hh>
#include <iostream>

int main() {
  namespace gh = glasshelix;
  gh::models::candidate_set candidates({{"a", "whole-a", {1, 2}, 1}, {"b", "whole-b", {8, 9}, 3}});
  gh::experiment::record record;
  record.experiment_id = "installed-contracts";
  record.source = {gh::data::source_kind::synthetic_fixture, "input", "audit", {"manifest"}};
  record.provenance.scientific_model_id = "model";
  record.provenance.model_revision = "1";
  record.provenance.gh_source_commit = "installed-gh-contract-fixture";
  record.provenance.ce_source_commit = "published-ce-header-fixture";
  record.provenance.numerical_policy = "contract-only";
  record.validate(candidates);
  if (candidates.values()[0].weight != .25 || candidates.values()[1].weight != .75 ||
      candidates.values()[0].joint_state != std::vector<double>{1, 2} ||
      candidates.values()[1].joint_state != std::vector<double>{8, 9}) return 1;
  std::cout << "installed actual GH/CE public contracts and whole alternatives passed\n";
}
