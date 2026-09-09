#include <GlassHelix/core/provenance.hh>
int main(){using namespace glasshelix::core;result_provenance p;p.scientific_model_id="supplied";p.model_revision="1";p.gh_source_commit="gh";p.ce_source_commit="ce";p.units={"hours"};p.scales={1};p.numerical_policy="fp32 mathematical derivative at stored values";
 p.numerical_approximations={{approximation_kind::bounded,"integration","fixture",.001}};p.inference_uncertainty={"measurement noise"};p.compatible_mechanisms={"a","b"};if(!p.valid())return 1;
 p.numerical_approximations[0].absolute_bound.reset();if(p.valid())return 2;return p.compatible_mechanisms.size()==2?0:3;
}
