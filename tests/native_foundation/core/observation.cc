#include <GlassHelix/core/observation.hh>
int main(){using namespace glasshelix::core;
 quantity q; q.axis={{identity::biological_abi_version,identity::serialized_record_kind::persistent_axis_identity,sizeof(identity::persistent_axis_identity)},{1,0},{2,0},{3,0},{4,0}};q.structure={5,0};q.units="counts";q.role=quantity_role::observation;q.extent=2;
 observation_record a{q,{0,99},{true,false},"reporter","hours","cell1","supplied",0,sampling_unit::individual,false};auto b=a;b.time=1;
 if(!a.valid()||!admissible_individual_successor(a,b)||!a.present[0]||a.present[1])return 1;
 b.destructive=true;if(admissible_individual_successor(a,b))return 2;
 inferential_state x{{{"h1",{1,2},1}}, {}, {0}},y=x;y.candidates[0].weight=2;
 return x.valid()&&y.valid()&&x.candidates[0].weight==1?0:3;
}
