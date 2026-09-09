#include <GlassHelix/core/host_bindings.hh>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
using namespace glasshelix::core;
void require(bool value) { if(!value) throw std::runtime_error("native binding contract failure"); }
quantity field(quantity_role role,std::size_t n) {
 quantity q;q.axis={{identity::biological_abi_version,identity::serialized_record_kind::persistent_axis_identity,sizeof(identity::persistent_axis_identity)},{1,0},{2,0},{3,0},{4,0}};
 q.structure={5,0};q.units="dimensionless";q.role=role;q.extent=n;return q;
}
template<class F> void rejects(F f){bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}require(rejected);}
int main() {
 auto definition=std::make_shared<const system_definition>("supplied-system",std::vector<quantity>{field(quantity_role::dynamical_state,2),field(quantity_role::parameter,1),field(quantity_role::observation,7)},std::vector<mechanism>{{"nonlinear",{0,1},{"state","parameter"},{0},false}});
 std::array<double,2> state{0,std::numeric_limits<double>::infinity()};std::array<double,1> parameters{3};
 auto a=bind_host_instance(definition,state,parameters),b=bind_host_instance(definition,state,parameters);
 state[0]=10;a.state[0]=20;a.parameters[0]=30;
 require(b.state[0]==0&&b.parameters[0]==3&&std::isinf(b.state[1]));
 require(a.definition==b.definition&&a.state_generation.value==0&&a.parameter_generation.value==0);
 auto dimensions=required_host_dimensions(*definition);require(dimensions.state==2&&dimensions.parameters==1);
 rejects([&]{bind_host_instance(definition,parameters,parameters);});
 rejects([&]{bind_host_instance(definition,state,state);});rejects([&]{bind_host_instance(nullptr,state,parameters);});
 auto empty=std::make_shared<const system_definition>("empty",std::vector<quantity>{},std::vector<mechanism>{});require(bind_host_instance(empty,{},{}).state.empty());
 system_definition oversized("overflow",{field(quantity_role::dynamical_state,std::numeric_limits<std::size_t>::max()),field(quantity_role::dynamical_state,1)},{});
 bool overflow=false;try{required_host_dimensions(oversized);}catch(const std::overflow_error&){overflow=true;}require(overflow);
}
