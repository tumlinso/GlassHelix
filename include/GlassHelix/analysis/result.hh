#pragma once

#include <GlassHelix/core/provenance.hh>
#include <stdexcept>

namespace glasshelix::analysis {

enum class assessment { not_run, qualified, unsupported };

// Claims are per action. Qualifying forward never implies a derivative or a
// biological result. Evidence references retain the exact consumer receipts.
struct capability_result {
  std::string action;
  assessment status = assessment::not_run;
  std::string evidence_id;

  void validate() const {
    if (action.empty() || evidence_id.empty() ||
        (status != assessment::not_run && status != assessment::qualified &&
         status != assessment::unsupported))
      throw std::invalid_argument("capability requires action, status and evidence");
  }
};

}  // namespace glasshelix::analysis
