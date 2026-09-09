#pragma once
#include <GlassHelix/core/quantity.hh>
namespace glasshelix::core {
enum class approximation_kind { exact_declared_arithmetic, bounded, empirical, unassessed };
struct approximation {
 approximation_kind kind=approximation_kind::unassessed;
 std::string source; // storage quantization, arithmetic, integration, pruning
 std::string applicable_domain;
 std::optional<double> absolute_bound;
};
struct result_provenance {
 std::string scientific_model_id, model_revision, gh_source_commit, ce_source_commit;
 identity::structure_epoch structure_epoch{};
 identity::value_generation state_generation{},parameter_generation{};
 std::vector<std::string> units;
 std::vector<double> scales;
 std::string numerical_policy;
 std::vector<std::string> model_assumptions, prior_constraints, empirical_evidence;
 std::vector<approximation> numerical_approximations;
 std::vector<std::string> inference_uncertainty, compatible_mechanisms, biological_labels;
 bool valid()const {
  if(scientific_model_id.empty()||model_revision.empty()||gh_source_commit.empty()||ce_source_commit.empty()||numerical_policy.empty()||units.size()!=scales.size())return false;
  for(std::size_t i=0;i<units.size();++i)if(units[i].empty()||!std::isfinite(scales[i])||scales[i]<=0)return false;
  for(const auto& a:numerical_approximations){
   if(a.source.empty()||a.applicable_domain.empty())return false;
   if(a.kind==approximation_kind::bounded&&!a.absolute_bound)return false;
   if(a.absolute_bound&&(!std::isfinite(*a.absolute_bound)||*a.absolute_bound<0))return false;
  }
  return true;
 }
};
}
