#pragma once
#include <GlassHelix/dynamics/evolution.hh>
#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <array>
#include <span>

namespace glasshelix::dynamics::response {
namespace df = cellerator::compute::differential;
namespace nf = df::nf1;

enum class primal_policy { saved, explicitly_recomputed };

struct saved_primal {
  cellerator::execution::value_generation state{};
  cellerator::execution::value_generation parameters{};
  std::uint64_t forcing = 0;
  std::uint64_t activity = 0;
  primal_policy policy = primal_policy::saved;
};

inline bool valid_attachment(const response_attachment& attachment,
                             forward_plan_identity actual,
                             void* stream) noexcept {
  const auto* block = attachment.block;
  const auto& owners = block ? block->device_primal : df::local_primal_owners{};
  return block && attachment.forward.stage_id && attachment.forward.candidate_id &&
      attachment.forward.stage_id == actual.stage_id &&
      attachment.forward.candidate_id == actual.candidate_id &&
      owners.left && owners.state && owners.parameters && owners.stream == stream &&
      cellerator::compute::operation::v2::same_stable_id(block->block.contract.definition, owners.identity.instance.prepared.definition);
}

inline bool current(const saved_primal& saved, const saved_primal& live) noexcept {
  return saved.policy == primal_policy::saved && live.policy == primal_policy::saved &&
      saved.state.value == live.state.value &&
      saved.parameters.value == live.parameters.value &&
      saved.forcing == live.forcing && saved.activity == live.activity;
}

inline bool valid_block(const df::local_block* block, void* stream) noexcept {
  if (!block) return false;
  const auto& owners = block->device_primal;
  return owners.left && owners.state && owners.parameters && owners.stream == stream &&
      cellerator::compute::operation::v2::same_stable_id(
          block->block.contract.definition, owners.identity.instance.prepared.definition);
}

inline program::program_status submit_response(
    const df::local_block& block, nf::capability action,
    forward_plan_identity identity, df::response_binding<float>& binding,
    void* stream) noexcept {
  program::prepared_stage_v2 stage{};
  if (df::make_local_device_stage(block, action, identity.stage_id,
                                  identity.candidate_id, 0, stage) != nf::status::success)
    return program::program_status::invalid_argument;
  const program::prepared_program_v2 p{2, 0, &stage, 1, nullptr, 0};
  const program::launch_binding_v2 launch{&binding};
  return program::execute_prepared_program_v2(p, &launch, 1, stream);
}

inline program::program_status linear_axpby(
    const cellerator::compute::native_numeric::resident_vector& input,
    const cellerator::compute::native_numeric::resident_vector& value,
    cellerator::compute::native_numeric::resident_vector& output,
    float alpha, float beta, void* stream) noexcept {
  using namespace cellerator::compute::native_numeric;
  linear_stage descriptor{linear_kind::axpby, input.elements, input.representation,
                          alpha, beta};
  const auto stage = make_linear_stage(0x47485250u, 1, &descriptor);
  return submit_one(stage, {input.data, output.data, value.data}, stream);
}

inline program::program_status execute_direct_map_jvp(
    core::differentiated_object requested, const direct_map_plan& plan,
    const saved_primal& saved, const saved_primal& live,
    df::response_binding<float>& binding, void* stream) noexcept {
  if (requested != plan.object ||
      !valid_attachment(plan.response, identity_of(plan), stream) || !current(saved, live))
    return program::program_status::launch_failed;
  return submit_response(*plan.response.block, nf::jvp, plan.response.forward,
                         binding, stream);
}

inline program::program_status execute_direct_map_vjp(
    core::differentiated_object requested, const direct_map_plan& plan,
    const saved_primal& saved, const saved_primal& live,
    df::response_binding<float>& binding, void* stream) noexcept {
  if (requested != plan.object ||
      !valid_attachment(plan.response, identity_of(plan), stream) || !current(saved, live))
    return program::program_status::launch_failed;
  return submit_response(*plan.response.block, nf::vjp, plan.response.forward,
                         binding, stream);
}

struct rk4_jvp_buffers {
  const cellerator::compute::native_numeric::resident_vector& state_direction;
  std::array<const cellerator::compute::native_numeric::resident_vector*, 4> forcing_direction;
  cellerator::compute::native_numeric::resident_vector& trial_direction;
  std::array<cellerator::compute::native_numeric::resident_vector*, 4> field_direction;
  cellerator::compute::native_numeric::resident_vector& packed;
  cellerator::compute::native_numeric::resident_vector& output;
};

inline program::program_status execute_rk4_jvp(
    const rk4_step_plan& plan, const saved_primal& saved, const saved_primal& live,
    std::array<df::response_binding<float>, 4>& bindings,
    rk4_jvp_buffers buffers, void* stream) noexcept {
  if (!current(saved, live) || stream != plan.stream() || !plan.identity().stage_id ||
      plan.field().response.forward.stage_id != plan.identity().stage_id ||
      plan.field().response.forward.candidate_id != plan.identity().candidate_id)
    return program::program_status::launch_failed;
  for (const auto* block : plan.field().response.rk4_blocks)
    if (!valid_block(block, stream)) return program::program_status::launch_failed;
  for (std::size_t i = 0; i < 4; ++i) {
    if (!buffers.forcing_direction[i] || !buffers.field_direction[i])
      return program::program_status::invalid_argument;
    bindings[i].left_direction = i == 0 ? &buffers.state_direction : &buffers.trial_direction;
    bindings[i].right_direction = buffers.forcing_direction[i];
    bindings[i].output = buffers.field_direction[i];
    if (auto status = submit_response(*plan.field().response.rk4_blocks[i], nf::jvp,
                                      plan.field().response.forward, bindings[i], stream);
        status != program::program_status::success) return status;
    if (i < 3) {
      const float coefficient = i < 2 ? .5f * plan.step() : plan.step();
      if (auto status = linear_axpby(buffers.state_direction, *buffers.field_direction[i],
                                    buffers.trial_direction, 1.f, coefficient, stream);
          status != program::program_status::success) return status;
    }
  }
  auto* packed = static_cast<std::byte*>(buffers.packed.data);
  const auto stride = buffers.state_direction.elements * sizeof(float);
  const std::array<const cellerator::compute::native_numeric::resident_vector*, 5> values{
      &buffers.state_direction, buffers.field_direction[0], buffers.field_direction[1],
      buffers.field_direction[2], buffers.field_direction[3]};
  for (std::size_t i = 0; i < values.size(); ++i)
    if (auto status = submit_one(plan.copy_stage(),
          {values[i]->data, packed + i * stride}, stream);
        status != program::program_status::success) return status;
  return submit_one(plan.combine().stage,
      {buffers.packed.data, buffers.output.data, nullptr, buffers.packed.data, 5 * stride}, stream);
}

struct rk4_vjp_buffers {
  const cellerator::compute::native_numeric::resident_vector& output_cotangent;
  const cellerator::compute::native_numeric::resident_vector& zero;
  cellerator::compute::native_numeric::resident_vector& state_adjoint;
  cellerator::compute::native_numeric::resident_vector& temporary;
  cellerator::compute::native_numeric::resident_vector& stage_state_adjoint;
  std::array<cellerator::compute::native_numeric::resident_vector*, 4> field_adjoint;
  std::array<cellerator::compute::native_numeric::resident_vector*, 4> forcing_adjoint;
};

inline program::program_status execute_rk4_vjp(
    const rk4_step_plan& plan, const saved_primal& saved, const saved_primal& live,
    std::array<df::response_binding<float>, 4>& bindings,
    rk4_vjp_buffers buffers, void* stream) noexcept {
  if (!current(saved, live) || stream != plan.stream() || !plan.identity().stage_id ||
      plan.field().response.forward.stage_id != plan.identity().stage_id ||
      plan.field().response.forward.candidate_id != plan.identity().candidate_id)
    return program::program_status::launch_failed;
  for (const auto* block : plan.field().response.rk4_blocks)
    if (!valid_block(block, stream)) return program::program_status::launch_failed;
  if (auto status = linear_axpby(buffers.zero, buffers.output_cotangent,
                                 buffers.state_adjoint, 1.f, 1.f, stream);
      status != program::program_status::success) return status;
  const std::array<float, 4> weights{plan.step()/6.f, plan.step()/3.f,
                                     plan.step()/3.f, plan.step()/6.f};
  for (std::size_t i = 0; i < 4; ++i) {
    if (!buffers.field_adjoint[i] || !buffers.forcing_adjoint[i])
      return program::program_status::invalid_argument;
    if (auto status = linear_axpby(buffers.zero, buffers.output_cotangent,
                                  *buffers.field_adjoint[i], 1.f, weights[i], stream);
        status != program::program_status::success) return status;
  }
  for (std::size_t reverse = 4; reverse-- > 0;) {
    bindings[reverse].cotangent = buffers.field_adjoint[reverse];
    bindings[reverse].left_adjoint = &buffers.stage_state_adjoint;
    bindings[reverse].right_adjoint = buffers.forcing_adjoint[reverse];
    if (auto status = submit_response(*plan.field().response.rk4_blocks[reverse], nf::vjp,
                                      plan.field().response.forward, bindings[reverse], stream);
        status != program::program_status::success) return status;
    if (auto status = linear_axpby(buffers.state_adjoint, buffers.stage_state_adjoint,
                                  buffers.temporary, 1.f, 1.f, stream);
        status != program::program_status::success) return status;
    if (auto status = linear_axpby(buffers.zero, buffers.temporary,
                                  buffers.state_adjoint, 1.f, 1.f, stream);
        status != program::program_status::success) return status;
    if (reverse > 0) {
      const float coefficient = reverse == 3 ? plan.step() : .5f * plan.step();
      if (auto status = linear_axpby(*buffers.field_adjoint[reverse-1],
                                    buffers.stage_state_adjoint, buffers.temporary,
                                    1.f, coefficient, stream);
          status != program::program_status::success) return status;
      if (auto status = linear_axpby(buffers.zero, buffers.temporary,
                                    *buffers.field_adjoint[reverse-1], 1.f, 1.f, stream);
          status != program::program_status::success) return status;
    }
  }
  return program::program_status::success;
}

inline program::program_status execute_observation_jvp(
    const direct_map_plan& plan, const saved_primal& saved, const saved_primal& live,
    df::response_binding<float>& binding, void* stream) noexcept {
  if (plan.object != core::differentiated_object::observation)
    return program::program_status::invalid_argument;
  return execute_direct_map_jvp(core::differentiated_object::observation,
                                plan, saved, live, binding, stream);
}

inline program::program_status execute_observation_vjp(
    const direct_map_plan& plan, const saved_primal& saved, const saved_primal& live,
    df::response_binding<float>& binding, void* stream) noexcept {
  return execute_direct_map_vjp(core::differentiated_object::observation,
                                plan, saved, live, binding, stream);
}

inline program::program_status execute_rollout_jvp(
    std::span<const direct_map_plan> plans, std::span<const saved_primal> saved,
    std::span<const saved_primal> live,
    std::span<df::response_binding<float>> bindings, void* stream) noexcept {
  if (plans.size() != saved.size() || saved.size() != live.size() ||
      saved.size() != bindings.size() || saved.empty())
    return program::program_status::invalid_argument;
  for (std::size_t i = 0; i < saved.size(); ++i) {
    if (plans[i].object != core::differentiated_object::implemented_rollout)
      return program::program_status::invalid_argument;
    if (auto status = execute_direct_map_jvp(
            core::differentiated_object::implemented_rollout,
            plans[i], saved[i], live[i], bindings[i], stream);
        status != program::program_status::success) return status;
  }
  return program::program_status::success;
}

struct rollout_vjp_buffers {
  const cellerator::compute::native_numeric::resident_vector& output_cotangent;
  std::span<cellerator::compute::native_numeric::resident_vector* const> state_adjoint;
  std::span<cellerator::compute::native_numeric::resident_vector* const> auxiliary_adjoint;
};

inline program::program_status execute_rollout_vjp(
    std::span<const direct_map_plan> plans, std::span<const saved_primal> saved,
    std::span<const saved_primal> live,
    std::span<df::response_binding<float>> bindings,
    rollout_vjp_buffers buffers, void* stream) noexcept {
  if (plans.size() != saved.size() || saved.size() != live.size() ||
      saved.size() != bindings.size() || saved.size() != buffers.state_adjoint.size() ||
      saved.size() != buffers.auxiliary_adjoint.size() || saved.empty())
    return program::program_status::invalid_argument;
  for (std::size_t reverse = plans.size(); reverse-- > 0;) {
    if (plans[reverse].object != core::differentiated_object::implemented_rollout ||
        !buffers.state_adjoint[reverse] || !buffers.auxiliary_adjoint[reverse])
      return program::program_status::invalid_argument;
    bindings[reverse].cotangent = reverse + 1 == plans.size()
        ? &buffers.output_cotangent : buffers.state_adjoint[reverse + 1];
    bindings[reverse].left_adjoint = buffers.state_adjoint[reverse];
    bindings[reverse].right_adjoint = buffers.auxiliary_adjoint[reverse];
    if (auto status = execute_direct_map_vjp(
            core::differentiated_object::implemented_rollout, plans[reverse],
            saved[reverse], live[reverse], bindings[reverse], stream);
        status != program::program_status::success) return status;
  }
  return program::program_status::success;
}
} // namespace glasshelix::dynamics::response
