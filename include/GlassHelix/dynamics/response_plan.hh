#pragma once
#include <GlassHelix/dynamics/evolution.hh>
#include <Cellerator/compute/operation/differential/local_arithmetic.hh>

namespace glasshelix::dynamics::response {
namespace df = cellerator::compute::differential;
namespace nf = df::nf1;

struct saved_primal {
  cellerator::execution::value_generation state{};
  cellerator::execution::value_generation parameters{};
  std::uint64_t forcing = 0;
  std::uint64_t activity = 0;
  bool recomputed = false;
};

inline bool valid_attachment(const response_attachment& attachment,
                             forward_plan_identity actual, void* stream) noexcept {
  const auto* block = attachment.block;
  const auto& owners = block ? block->device_primal : df::local_primal_owners{};
  return block && attachment.forward.stage_id && attachment.forward.candidate_id &&
      attachment.forward.stage_id == actual.stage_id &&
      attachment.forward.candidate_id == actual.candidate_id &&
      owners.left && owners.state && owners.parameters && owners.stream == stream &&
      cellerator::compute::operation::v2::same_stable_id(block->block.contract.definition, owners.identity.instance.prepared.definition);
}

inline program::program_status execute_direct_map_jvp(
    const direct_map_plan& plan, const saved_primal& saved, const saved_primal& live,
    df::response_binding<float>& binding, void* stream) noexcept {
  if (!valid_attachment(plan.response, identity_of(plan), stream) ||
      saved.state.value != live.state.value || saved.parameters.value != live.parameters.value ||
      saved.forcing != live.forcing || saved.activity != live.activity) return program::program_status::launch_failed;
  program::prepared_stage_v2 stage{};
  if (df::make_local_device_stage(*plan.response.block, nf::jvp,
                                  plan.response.forward.stage_id, plan.response.forward.candidate_id,
                                  0, stage) != nf::status::success) return program::program_status::invalid_argument;
  const program::prepared_program_v2 p{2,0,&stage,1,nullptr,0};
  const program::launch_binding_v2 launch{&binding};
  return program::execute_prepared_program_v2(p,&launch,1,stream);
}

inline program::program_status execute_direct_map_vjp(
    const direct_map_plan& plan, const saved_primal& saved, const saved_primal& live,
    df::response_binding<float>& binding, void* stream) noexcept {
  if (!valid_attachment(plan.response, identity_of(plan), stream) ||
      saved.state.value != live.state.value || saved.parameters.value != live.parameters.value ||
      saved.forcing != live.forcing || saved.activity != live.activity) return program::program_status::launch_failed;
  program::prepared_stage_v2 stage{};
  if (df::make_local_device_stage(*plan.response.block, nf::vjp,
                                  plan.response.forward.stage_id, plan.response.forward.candidate_id,
                                  0, stage) != nf::status::success) return program::program_status::invalid_argument;
  const program::prepared_program_v2 p{2,0,&stage,1,nullptr,0};
  const program::launch_binding_v2 launch{&binding};
  return program::execute_prepared_program_v2(p,&launch,1,stream);
}
} // namespace glasshelix::dynamics::response
