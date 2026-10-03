#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace glasshelix::data {

enum class source_kind { synthetic_fixture, audited_celltag };

// References an owner's audit and input manifest; it does not perform an audit
// or infer observed histories from population snapshots.
struct source_reference {
  source_kind kind = source_kind::synthetic_fixture;
  std::string source_id;
  std::string audit_evidence_id;
  std::vector<std::string> input_manifest_ids;

  void validate() const {
    if (source_id.empty() || audit_evidence_id.empty() || input_manifest_ids.empty())
      throw std::invalid_argument("source, audit and input manifest identities required");
    for (const auto& id : input_manifest_ids)
      if (id.empty()) throw std::invalid_argument("empty input manifest identity");
    if (kind != source_kind::synthetic_fixture && kind != source_kind::audited_celltag)
      throw std::invalid_argument("unknown source kind");
  }
};

}  // namespace glasshelix::data
