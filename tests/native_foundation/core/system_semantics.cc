#include <GlassHelix/core/system.hh>
#include <stdexcept>
using namespace glasshelix::core;
void require(bool x){if(!x)throw std::runtime_error("n-ary semantics failure");}
quantity q(){quantity x;x.axis={{identity::biological_abi_version,identity::serialized_record_kind::persistent_axis_identity,sizeof(identity::persistent_axis_identity)},{1,0},{2,0},{3,0},{4,0}};x.structure={5,0};x.units="dimensionless";x.extent=1;return x;}
int main(){
 mechanism m{"interaction",{0,1,2},{"substrate","modulator","context"},{3},false};
 auto def=std::make_shared<const system_definition>("supplied-nary",std::vector<quantity>(4,q()),std::vector<mechanism>{m});
 system_instance a{def,{1,2,3,0},{1},{0},{0}},b{def,{4,5,6,0},{2},{0},{0}};
 a.state[0]=9;require(b.state[0]==4&&a.definition==b.definition);
 auto repeated=m;repeated.arguments={0,0,2};bool rejected=false;
 try{system_definition invalid("bad",std::vector<quantity>(4,q()),{repeated});}catch(const std::invalid_argument&){rejected=true;}require(rejected);
 repeated.repeated_arguments=true;system_definition admitted("ok",std::vector<quantity>(4,q()),{repeated});
 auto badrole=m;badrole.argument_roles.pop_back();rejected=false;
 try{system_definition invalid("bad",std::vector<quantity>(4,q()),{badrole});}catch(const std::invalid_argument&){rejected=true;}require(rejected);
 std::vector<realization_link> split{{"interaction",{"stage1","stage2"}}};
 std::vector<realization_link> fused{{"interaction",{"fused-stage"}},{"other-mechanism",{"fused-stage"}}};
 require(split[0].mechanism_id==fused[0].mechanism_id&&def->mechanisms()[0].arguments.size()==3);
 require(def->mechanisms()[0].scientific_id=="interaction"&&admitted.mechanisms()[0].repeated_arguments);
}
