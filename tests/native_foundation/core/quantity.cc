#include <GlassHelix/core/quantity.hh>
#include <stdexcept>
using namespace glasshelix::core;
void require(bool x) { if(!x) throw std::runtime_error("quantity contract violation"); }
int main() {
 quantity a; a.axis={{identity::biological_abi_version,identity::serialized_record_kind::persistent_axis_identity,sizeof(identity::persistent_axis_identity)},{1,0},{2,0},{3,0},{4,0}};
 a.structure={5,0};a.units="dimensionless";a.extent=33;
 require(valid(a));require(!a.biological_label);auto b=a;require(interchangeable(a,b));
 b.axis.domain.low++;require(!interchangeable(a,b));b=a;b.axis.order.low++;require(!interchangeable(a,b));
 b=a;b.role=quantity_role::observation;require(!interchangeable(a,b));b=a;b.biological_label="optional";require(interchangeable(a,b));
 b=a;b.scale=0;require(!valid(b));
}
