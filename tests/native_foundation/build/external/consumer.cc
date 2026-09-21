#include <GlassHelix/system/assembly.hh>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ix = cellerator::compute::operation::indexed;
namespace pg = cellerator::execution::program;
namespace {
constexpr std::size_t width=33;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void cuda_check(cudaError_t status,const char* why){check(status==cudaSuccess,why);}
glasshelix::core::quantity field(){using namespace glasshelix::core; quantity q;
 q.axis={{identity::biological_abi_version,identity::serialized_record_kind::persistent_axis_identity,sizeof(identity::persistent_axis_identity)},{1,0},{2,0},{3,0},{4,0}};
 q.structure={5,0};q.units="dimensionless";q.extent=width;return q;}
}
int main() try {
 namespace gh=glasshelix;
 auto definition=std::make_shared<const gh::core::system_definition>("external-width33",std::vector<gh::core::quantity>(4,field()),std::vector<gh::core::mechanism>{{"forcing-aware-triad",{0,1,2},{"left","right","forcing"},{3},false}});
 gh::system::compiled_implementation impl;impl.numerical_definition={10,0};impl.assembly_owner={11,0};impl.input_roles={{1,0},{2,0},{3,0}};impl.output_roles={{4,0}};impl.output_effects={gh::system::nf::output_signature{}.effect};
 using nt=cellerator::execution::numeric_type;impl.numeric.relation_storage=nt::f32;impl.numeric.state_storage=nt::f32;impl.numeric.multiply=nt::f32;impl.numeric.accumulation=nt::f32;impl.numeric.output_storage=nt::f32;impl.numeric.scalar=nt::f32;
 // No GH callback: Cellerator owns the registered evaluator's CUDA launch.
 gh::system::assembled_mechanism mechanism(definition,0,impl);
 ix::block_registry registry;const ix::registered_block custom{{21,0},ix::evaluator_opcode::first_minus_product_tail,3,true,true,true};
 check(registry.register_block(custom)==ix::evaluation_status::success,"registered custom block rejected");
 const ix::prepared_evaluator_stage evaluator{custom,3,7};std::array<ix::evaluator_stage_values,width> values{};
 std::vector<pg::prepared_stage_v2> stages;std::vector<pg::launch_binding_v2> bindings;stages.reserve(width);bindings.reserve(width);
 std::array<float,width*3> input{};
 for(std::size_t i=0;i<width;++i){input[3*i]=.01f*float(i);input[3*i+1]=.2f+.03f*float(i);input[3*i+2]=i<16?.5f:0.f;values[i].value_generation=7;stages.push_back(mechanism.bind_indexed_stage(evaluator,100+i,1,unsigned(i)));}
 float* device_input=nullptr;float* device_output=nullptr;cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_input),sizeof(input)),"input allocation failed");cuda_check(cudaMalloc(reinterpret_cast<void**>(&device_output),width*sizeof(float)),"output allocation failed");cuda_check(cudaMemcpy(device_input,input.data(),sizeof(input),cudaMemcpyHostToDevice),"input upload failed");
 for(std::size_t i=0;i<width;++i)bindings.push_back({device_input+3*i,device_output+i,&values[i]});
 const pg::prepared_program_v2 program{2,0,stages.data(),stages.size(),nullptr,0};check(pg::execute_prepared_program_v2(program,bindings.data(),bindings.size(),nullptr)==pg::program_status::success,"registered evaluator launch failed");cuda_check(cudaDeviceSynchronize(),"registered evaluator synchronization failed");
 std::array<float,width> output{};cuda_check(cudaMemcpy(output.data(),device_output,sizeof(output),cudaMemcpyDeviceToHost),"output download failed");cudaFree(device_output);cudaFree(device_input);
 for(std::size_t i=0;i<width;++i){const auto expected=input[3*i]-input[3*i+1]*input[3*i+2];check(std::abs(output[i]-expected)<1e-6f,"width33 custom/composed parity failed");}
 return 0;
} catch(const std::exception&) {return 1;}
