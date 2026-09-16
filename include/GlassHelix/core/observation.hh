#pragma once
#include <GlassHelix/core/quantity.hh>
#include <stdexcept>
namespace glasshelix::core {
enum class sampling_unit { individual, population_snapshot };
struct observation_record { quantity measured; std::vector<double> values; std::vector<bool> present; std::string modality, time_units, sampling_id, assumptions; double time=0; sampling_unit unit=sampling_unit::population_snapshot; bool destructive=true; bool valid()const{return core::valid(measured)&&measured.role==quantity_role::observation&&values.size()==measured.extent&&present.size()==values.size()&&!modality.empty()&&!time_units.empty()&&!sampling_id.empty()&&!assumptions.empty()&&std::isfinite(time);} };
inline bool admissible_individual_successor(const observation_record& a,const observation_record& b){return a.valid()&&b.valid()&&a.unit==sampling_unit::individual&&b.unit==sampling_unit::individual&&!a.destructive&&!b.destructive&&a.sampling_id==b.sampling_id&&a.time_units==b.time_units&&a.time<b.time;}
struct supplied_candidate { std::string hypothesis_id; std::vector<double> joint_coordinates; double weight=0; };
struct inferential_state { std::vector<supplied_candidate> candidates; std::vector<std::string> evidence_ids; identity::value_generation generation{}; bool valid()const{double total=0;for(const auto& c:candidates){if(c.hypothesis_id.empty()||!std::isfinite(c.weight)||c.weight<0)return false;total+=c.weight;}return !candidates.empty()&&std::isfinite(total)&&total>0;} };
}
