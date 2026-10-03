#pragma once

#include <GlassHelix/inference/inference.hh>

namespace glasshelix::models {

// The preferred model seam is the existing atomic joint-candidate owner.
// Adapters retain hypothesis, mechanism and complete joint state together.
using candidate_set = inference::finite_candidates;
using joint_candidate = inference::candidate;

}  // namespace glasshelix::models
