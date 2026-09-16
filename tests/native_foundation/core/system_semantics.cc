#include <GlassHelix/core/system.hh>
#include <stdexcept>
using namespace glasshelix::core;
quantity field(){ quantity q; q.axis={{identity::biological_abi_version,identity::serialized_record_kind::persistent_axis_identity,sizeof(identity::persistent_axis_identity)},{1,0},{2,0},{3,0},{4,0}}; q.structure={5,0}; q.units="dimensionless"; q.extent=1; return q; }
int main() {
  mechanism interaction{"interaction",{0,1,2},{"substrate","modulator","context"},{3},false};
  auto definition=std::make_shared<const system_definition>("supplied-nary",std::vector<quantity>(4,field()),std::vector<mechanism>{interaction});
  auto repeated=interaction; repeated.arguments={0,0,2}; bool rejected=false;
  try { system_definition invalid("bad",std::vector<quantity>(4,field()),{repeated}); } catch(const std::invalid_argument&) { rejected=true; }
  if(!rejected) return 1; repeated.repeated_arguments=true;
  system_definition admitted("ok",std::vector<quantity>(4,field()),{repeated});
  return definition->mechanisms()[0].arguments.size()==3 && admitted.mechanisms()[0].repeated_arguments ? 0 : 2;
}
