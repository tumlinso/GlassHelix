#include <GlassHelix/system/assembly.hh>
#include <array>
#include <cmath>
#include <stdexcept>
using namespace glasshelix;
namespace pg=cellerator::execution::program;
namespace nf=cellerator::compute::operation::nf1;
struct inputs { std::array<double,33> a,b,c; };
pg::program_status triad(const void*,const pg::launch_binding_v2& binding,void*)noexcept {
 if(!binding.input||!binding.output)return pg::program_status::invalid_argument;
 const auto& x=*static_cast<const inputs*>(binding.input);auto* y=static_cast<double*>(binding.output);
 for(std::size_t i=0;i<33;++i)y[i]=std::tanh(x.a[i]*x.b[i]+x.c[i]*x.c[i]);
 return pg::program_status::success;
}
void require(bool x){if(!x)throw std::runtime_error("scientific assembly failure");}
core::quantity field(){core::quantity q;q.axis={{core::identity::biological_abi_version,core::identity::serialized_record_kind::persistent_axis_identity,sizeof(core::identity::persistent_axis_identity)},{1,0},{2,0},{3,0},{4,0}};q.structure={5,0};q.units="dimensionless";q.extent=33;return q;}
int main(){
 auto scientific=std::make_shared<const core::system_definition>("supplied",std::vector<core::quantity>(4,field()),std::vector<core::mechanism>{{"joint-triad",{0,1,2},{"first","second","third"},{3},false}});
 system::compiled_implementation impl;impl.numerical_definition={10,0};impl.assembly_owner={11,0};impl.input_roles={{1,0},{2,0},{3,0}};impl.output_roles={{4,0}};impl.output_effects={nf::output_signature{}.effect};impl.forward=triad;
 using nt=cellerator::execution::numeric_type;impl.numeric.relation_storage=nt::f64;impl.numeric.state_storage=nt::f64;impl.numeric.multiply=nt::f64;impl.numeric.accumulation=nt::f64;impl.numeric.output_storage=nt::f64;impl.numeric.scalar=nt::f64;
 auto m=std::make_shared<const system::assembled_mechanism>(scientific,0,impl);require(m->scientific_mechanism().arguments.size()==3);require(m->scientific_mechanism().scientific_id=="joint-triad");
 pg::prepared_stage_v2 stage;require(m->bind_stage(nf::forward,nullptr,1,1,0,stage)==nf::status::success);require(m->bind_stage(nf::jvp,nullptr,2,1,0,stage)==nf::status::unsupported_derivative);
 require(m->bind_stage(nf::forward,nullptr,7,2,0,stage)==nf::status::success);require(m->scientific_mechanism().scientific_id=="joint-triad");
 inputs x;for(std::size_t i=0;i<33;++i){x.a[i]=.01*i;x.b[i]=.2+i*.03;x.c[i]=-.1+i*.001;}
 std::array<double,33> y{};pg::launch_binding_v2 binding{&x,y.data()};pg::prepared_program_v2 program;program.stages=&stage;program.stage_count=1;
 require(pg::execute_prepared_program_v2(program,&binding,1,nullptr)==pg::program_status::success);
 for(std::size_t i=0;i<33;++i)require(std::abs(y[i]-std::tanh(x.a[i]*x.b[i]+x.c[i]*x.c[i]))<1e-14);
 // Nonzero mixed difference distinguishes this joint mechanism from additive pairs.
 auto f=[](double a,double b){return std::tanh(a*b+.04);};require(std::abs(f(.2,.3)-f(.2,0)-f(0,.3)+f(0,0))>.01);
 bool rejected=false;impl.input_roles.pop_back();try{system::assembled_mechanism bad(scientific,0,impl);}catch(const std::invalid_argument&){rejected=true;}require(rejected);
}
