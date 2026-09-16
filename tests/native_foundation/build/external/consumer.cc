#include <GlassHelix/system/assembly.hh>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
namespace pg = cellerator::execution::program;
namespace nf = cellerator::compute::operation::nf1;
struct inputs { std::array<float,33> left,right,forcing; };
pg::program_status nonlinear_map(const void*, const pg::launch_binding_v2& binding, void*) noexcept {
  if (!binding.input || !binding.output) return pg::program_status::invalid_argument;
  const auto& x=*static_cast<const inputs*>(binding.input); auto* out=static_cast<float*>(binding.output);
  for (std::size_t i=0;i<33;++i) out[i]=std::tanh(x.left[i]*x.right[i]+x.forcing[i]);
  return pg::program_status::success;
}
glasshelix::core::quantity field_quantity() {
  using namespace glasshelix::core;
  glasshelix::core::quantity q; q.axis={{identity::biological_abi_version,identity::serialized_record_kind::persistent_axis_identity,sizeof(identity::persistent_axis_identity)},{1,0},{2,0},{3,0},{4,0}};
  q.structure={5,0}; q.units="dimensionless"; q.extent=33; return q;
}
int main() {
  namespace gh = glasshelix;
  auto definition=std::make_shared<const gh::core::system_definition>("external-width33",std::vector<gh::core::quantity>(4,field_quantity()),std::vector<gh::core::mechanism>{{"forcing-aware-triad",{0,1,2},{"left","right","forcing"},{3},false}});
  gh::system::compiled_implementation implementation; implementation.numerical_definition={10,0}; implementation.assembly_owner={11,0}; implementation.input_roles={{1,0},{2,0},{3,0}}; implementation.output_roles={{4,0}}; implementation.output_effects={nf::output_signature{}.effect}; implementation.forward=nonlinear_map;
  using numeric_type=cellerator::execution::numeric_type;
  implementation.numeric.relation_storage=numeric_type::f32; implementation.numeric.state_storage=numeric_type::f32; implementation.numeric.multiply=numeric_type::f32; implementation.numeric.accumulation=numeric_type::f32; implementation.numeric.output_storage=numeric_type::f32; implementation.numeric.scalar=numeric_type::f32;
  gh::system::assembled_mechanism mechanism(definition,0,implementation);
  pg::prepared_stage_v2 stage; if(mechanism.bind_stage(nf::forward,nullptr,1,1,0,stage)!=nf::status::success) return 1;
  inputs values{}; for(std::size_t i=0;i<33;++i){ values.left[i]=.01f*float(i); values.right[i]=.2f+.03f*float(i); values.forcing[i]=i<16?.5f:0.f; }
  std::array<float,33> output{}; pg::launch_binding_v2 binding{&values,output.data()}; pg::prepared_program_v2 program; program.stages=&stage; program.stage_count=1;
  if(pg::execute_prepared_program_v2(program,&binding,1,nullptr)!=pg::program_status::success) return 2;
  for(std::size_t i=0;i<33;++i) if(std::abs(output[i]-std::tanh(values.left[i]*values.right[i]+values.forcing[i]))>1e-6f) return 3;
}
