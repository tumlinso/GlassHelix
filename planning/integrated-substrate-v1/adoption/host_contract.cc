#include <GlassHelix/experiment/record.hh>
#include <iostream>

using namespace glasshelix;

void require(bool condition) {
  if (!condition) throw std::runtime_error("experiment foundation check failed");
}
template<class F> void rejected(F action) {
  bool caught = false;
  try { action(); } catch (const std::invalid_argument&) { caught = true; }
  require(caught);
}

int main() {
  models::candidate_set candidates({{"hypothesis-a", "mechanism-a", {1, 2}, 1},
                                     {"hypothesis-b", "mechanism-b", {3, 4}, 3}});
  experiment::record record;
  record.experiment_id = "host-fixture";
  record.source = {data::source_kind::synthetic_fixture, "fixture-inputs",
                   "fixture-audit", {"fixture-manifest"}};
  record.provenance.scientific_model_id = "joint-alternatives";
  record.provenance.model_revision = "1";
  record.provenance.gh_source_commit = "host-fixture-gh";
  record.provenance.ce_source_commit = "host-fixture-ce";
  record.provenance.numerical_policy = "metadata-only";
  record.capabilities = {{"forward", analysis::assessment::not_run, "host-scope"},
                         {"second_response", analysis::assessment::unsupported, "host-scope"},
                         {"biological_fit", analysis::assessment::not_run, "fixture-only"}};
  record.validate(candidates);
  require(candidates.values()[0].joint_state == std::vector<double>({1, 2}));
  require(candidates.values()[1].mechanism_id == "mechanism-b");
  require(candidates.values()[0].weight == .25 && candidates.values()[1].weight == .75);
  record.capabilities.back().status = analysis::assessment::qualified;
  rejected([&] { record.validate(candidates); });
  record.source.kind = data::source_kind::audited_celltag;
  record.validate(candidates);  // Admission of external evidence IDs, not a fit.
  record.source.audit_evidence_id.clear();
  rejected([&] { record.validate(candidates); });
  record.source.audit_evidence_id = "external-audit";
  record.capabilities.push_back(record.capabilities.front());
  rejected([&] { record.validate(candidates); });
  rejected([] { models::candidate_set invalid({{"h", "m", {1}, -1}}); });
  std::cout << "metadata identities, whole candidates and claim misuse passed\n";
}
