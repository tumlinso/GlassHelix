#pragma once
#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <GlassHelix/core/request.hh>
#include <cstdint>
#include <stdexcept>
namespace glasshelix::interrogation {
namespace df=cellerator::compute::differential; namespace nf=df::nf1; namespace pg=cellerator::execution::program;
enum class primal_policy { saved, explicitly_recomputed };
struct vector_field_primal { std::uint64_t state, parameters, forcing, activity; primal_policy policy; };
inline void require_current(const vector_field_primal& saved,const vector_field_primal& live){if(saved.state!=live.state||saved.parameters!=live.parameters||saved.forcing!=live.forcing||saved.activity!=live.activity)throw std::invalid_argument("stale vector-field primal");}
inline pg::program_status execute_vector_field_jvp(const core::response_request& request,const vector_field_primal& saved,const vector_field_primal& live,const df::local_block& block,df::response_binding<float>& binding,void* stream){if(request.object!=core::differentiated_object::vector_field)return pg::program_status::invalid_argument;try{require_current(saved,live);}catch(...){return pg::program_status::launch_failed;}pg::prepared_stage_v2 stage{};if(df::make_local_device_stage(block,nf::jvp,1,1,0,stage)!=nf::status::success)return pg::program_status::invalid_argument;const pg::prepared_program_v2 program{2,0,&stage,1,nullptr,0};const pg::launch_binding_v2 launch{&binding};return pg::execute_prepared_program_v2(program,&launch,1,stream);}
}
