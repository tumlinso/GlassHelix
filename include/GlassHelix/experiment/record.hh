#pragma once

#include <GlassHelix/analysis/result.hh>
#include <GlassHelix/data/source.hh>
#include <GlassHelix/models/candidate_set.hh>
#include <unordered_set>

namespace glasshelix::experiment {

// Common receipt envelope for existing native and Torch clients. Execution,
// fitted values, evidence permissions and audit logic retain their owners.
struct record {
  std::string experiment_id;
  data::source_reference source;
  core::result_provenance provenance;
  std::vector<analysis::capability_result> capabilities;

  void validate(const models::candidate_set& candidates) const {
    source.validate();
    if (experiment_id.empty() || !provenance.valid() || candidates.values().empty())
      throw std::invalid_argument("experiment identity, provenance and candidates required");
    std::unordered_set<std::string> actions;
    for (const auto& result : capabilities) {
      result.validate();
      if (!actions.insert(result.action).second)
        throw std::invalid_argument("duplicate capability action");
      if (result.action == "biological_fit" &&
          result.status == analysis::assessment::qualified &&
          source.kind != data::source_kind::audited_celltag)
        throw std::invalid_argument("biological fit requires audited CellTag inputs");
    }
  }
};

}  // namespace glasshelix::experiment
