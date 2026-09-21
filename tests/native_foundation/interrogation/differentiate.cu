#include <GlassHelix/dynamics/response_plan.hh>
#include <GlassHelix/interrogation/diagnostics.hh>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace gh = glasshelix;
namespace rsp = gh::dynamics::response;
namespace df = cellerator::compute::differential;
namespace nf = df::nf1;
namespace nn = cellerator::compute::native_numeric;
namespace ex = cellerator::execution;
namespace pg = ex::program;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void cuda_require(cudaError_t value, const char* message) {
  require(value == cudaSuccess, message);
}

struct vector {
  nn::resident_vector value{};
  vector(std::size_t count, cudaStream_t stream, float fill, std::uint64_t generation = 1) {
    cuda_require(nn::allocate(&value, count, nn::device_representation::f32, 0), "allocate");
    std::vector<float> host(count, fill);
    cuda_require(nn::upload(value, host.data(), count, {generation}, stream), "upload");
  }
  ~vector() { (void)nn::release(&value); }
  vector(const vector&) = delete;
};

std::vector<float> download(const vector& value, cudaStream_t stream) {
  std::vector<float> host(value.value.elements);
  cuda_require(nn::download(value.value, host.data(), host.size(), stream), "download");
  cuda_require(cudaStreamSynchronize(stream), "synchronize");
  return host;
}

nf::primal_record primal(nf::identity definition, std::uint64_t state_generation = 7,
                         std::uint64_t parameter_generation = 11) {
  nf::primal_record result{};
  result.instance.prepared = {definition, {2, 1}, {1}, {3, 1}};
  result.instance.state = {{4, 1}, {state_generation}};
  result.instance.parameters = {{5, 1}, {parameter_generation}};
  return result;
}

struct primitive {
  std::array<nf::operand_signature, 2> inputs{};
  nf::output_signature output{};
  nf::primal_record saved{};
  df::local_block block{};

  primitive(std::uint64_t definition, std::size_t count, cudaStream_t stream,
            const nn::resident_vector& left, const nn::resident_vector& right,
            const nn::resident_vector& state, const nn::resident_vector& parameters,
            nn::local_operation operation = nn::local_operation::add) {
    inputs = {{{{10, definition}, {}, count, ex::numeric_type::f32},
               {{11, definition}, {}, count, ex::numeric_type::f32}}};
    output = {{{12, definition}, {}, count, ex::numeric_type::f32}, {13, definition}};
    nf::operation_contract contract{};
    contract.definition = {definition, 1};
    const bool unary = operation == nn::local_operation::tanh;
    contract.arguments = {inputs.data(), unary ? 1u : 2u};
    contract.outputs = {&output, 1};
    contract.numeric = {ex::numeric_type::f32, ex::numeric_type::f32,
                        ex::numeric_type::f32, ex::numeric_type::f32,
                        ex::numeric_type::f32, ex::numeric_type::f32};
    contract.capabilities = nf::forward | nf::jvp | nf::vjp | nf::second_direction;
    saved = primal(contract.definition, state.generation.value, parameters.generation.value);
    df::local_primal_owners owners{&left, unary ? nullptr : &right,
                                   &state, &parameters, stream, saved};
    require(df::make_local_device_block(operation, contract, owners, block)
                == nf::status::success,
            "prepare response block");
  }

  df::response_binding<float> binding(
      nf::capability action, const nn::resident_vector* left_direction,
      const nn::resident_vector* right_direction, const nn::resident_vector* cotangent,
      const nn::resident_vector* result, const nn::resident_vector* left_adjoint,
      const nn::resident_vector* right_adjoint) const {
    df::response_binding<float> value{};
    value.left_direction = left_direction;
    value.right_direction = right_direction;
    value.cotangent = cotangent;
    value.output = result;
    value.left_adjoint = left_adjoint;
    value.right_adjoint = right_adjoint;
    value.count = block.block.contract.arguments[0].element_count;
    value.request.action = action;
    value.request.primal = saved;
    value.request.direction_domain = action == nf::vjp ? output.operand : inputs[0];
    value.request.response_domain = action == nf::vjp ? inputs[0] : output.operand;
    value.direction = value.request.direction_domain;
    value.response = value.request.response_domain;
    return value;
  }
};

gh::dynamics::prepared_stage_bundle forward_add_bundle(
    nn::linear_stage& descriptor, pg::prepared_stage_v2& stage,
    std::uint64_t stage_id, std::uint64_t candidate_id, std::size_t count) {
  descriptor = {nn::linear_kind::axpby, count, nn::device_representation::f32, 1.f, 1.f};
  stage = nn::make_linear_stage(stage_id, candidate_id, &descriptor);
  return {{2, 0, &stage, 1, nullptr, 0}, {pg::launch_binding_v2{}}};
}

struct multiply_forward_state { std::uint64_t count = 0; };
__global__ void multiply_forward_kernel(const float* state, const float* forcing,
                                        float* output, std::uint64_t count) {
  const auto index = std::uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (index < count) output[index] = state[index]*forcing[index];
}
pg::program_status admit_multiply_forward(
    const void* prepared, const pg::launch_binding_v2& binding, void* stream) noexcept {
  const auto* state = static_cast<const multiply_forward_state*>(prepared);
  return state && state->count && binding.input && binding.values && binding.output && stream
      ? pg::program_status::success : pg::program_status::invalid_argument;
}
pg::program_status launch_multiply_forward(
    const void* prepared, const pg::launch_binding_v2& binding, void* stream) noexcept {
  if (admit_multiply_forward(prepared, binding, stream) != pg::program_status::success)
    return pg::program_status::invalid_argument;
  const auto count = static_cast<const multiply_forward_state*>(prepared)->count;
  multiply_forward_kernel<<<static_cast<unsigned>((count+255)/256), 256, 0,
                            static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(binding.input), static_cast<const float*>(binding.values),
      static_cast<float*>(binding.output), count);
  return cudaGetLastError() == cudaSuccess ? pg::program_status::success
                                           : pg::program_status::launch_failed;
}

gh::dynamics::prepared_stage_bundle forward_multiply_bundle(
    multiply_forward_state& descriptor, pg::prepared_stage_v2& stage,
    std::uint64_t stage_id, std::uint64_t candidate_id, std::size_t count) {
  descriptor.count = count;
  stage = {stage_id, candidate_id, &descriptor, launch_multiply_forward,
           0, 0, 0, 0, admit_multiply_forward};
  return {{2, 0, &stage, 1, nullptr, 0}, {pg::launch_binding_v2{}}};
}

__global__ void tanh_forward_kernel(const float* state, float* output,
                                    std::uint64_t count) {
  const auto index = std::uint64_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (index < count) output[index] = tanhf(state[index]);
}
pg::program_status admit_tanh_forward(
    const void* prepared, const pg::launch_binding_v2& binding, void* stream) noexcept {
  const auto* state = static_cast<const multiply_forward_state*>(prepared);
  return state && state->count && binding.input && binding.output && stream
      ? pg::program_status::success : pg::program_status::invalid_argument;
}
pg::program_status launch_tanh_forward(
    const void* prepared, const pg::launch_binding_v2& binding, void* stream) noexcept {
  if (admit_tanh_forward(prepared, binding, stream) != pg::program_status::success)
    return pg::program_status::invalid_argument;
  const auto count = static_cast<const multiply_forward_state*>(prepared)->count;
  tanh_forward_kernel<<<static_cast<unsigned>((count+255)/256), 256, 0,
                        static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float*>(binding.input), static_cast<float*>(binding.output), count);
  return cudaGetLastError() == cudaSuccess ? pg::program_status::success
                                           : pg::program_status::launch_failed;
}
gh::dynamics::prepared_stage_bundle forward_tanh_bundle(
    multiply_forward_state& descriptor, pg::prepared_stage_v2& stage,
    std::uint64_t stage_id, std::uint64_t candidate_id, std::size_t count) {
  descriptor.count = count;
  stage = {stage_id, candidate_id, &descriptor, launch_tanh_forward,
           0, 0, 0, 0, admit_tanh_forward};
  return {{2, 0, &stage, 1, nullptr, 0}, {pg::launch_binding_v2{}}};
}

int main() {
  cudaStream_t stream{};
  cuda_require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
  constexpr std::size_t count = 33;
  constexpr std::uint64_t stage_id = 77, candidate_id = 9;

  vector state(count, stream, 2.f, 7), forcing(count, stream, .5f, 13);
  vector parameters(count, stream, 1.f, 11);
  ex::value_generation activity_generation{17};
  vector state_direction(count, stream, .2f), zero_direction(count, stream, 0.f);
  vector one_direction(count, stream, 1.f);
  vector forcing_direction(count, stream, .1f);
  vector direct_output(count, stream, 0.f), direct_left_adjoint(count, stream, 0.f);
  vector direct_right_adjoint(count, stream, 0.f), cotangent(count, stream, .3f);
  primitive direct_primitive(100, count, stream, state.value, forcing.value,
                             state.value, parameters.value);
  nn::linear_stage direct_descriptor{};
  pg::prepared_stage_v2 direct_stage{};
  auto direct_bundle = forward_add_bundle(direct_descriptor, direct_stage, stage_id,
                                          candidate_id, count);
  gh::dynamics::response_attachment direct_attachment{
      {stage_id, candidate_id}, &direct_primitive.block, {}, {&forcing.value},
      &activity_generation};
  gh::dynamics::direct_map_plan direct_plan{
      direct_bundle, gh::core::differentiated_object::vector_field, direct_attachment};
  rsp::saved_primal stamp = rsp::retain_primal(direct_attachment);

  auto direct_jvp = direct_primitive.binding(
      nf::jvp, &state_direction.value, &zero_direction.value, nullptr,
      &direct_output.value, nullptr, nullptr);
  require(rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                      direct_plan, stamp, direct_jvp, stream)
              == pg::program_status::success,
          "direct map JVP");
  for (float value : download(direct_output, stream))
    require(std::abs(value - .2f) < 2e-6f, "direct analytic JVP");
  auto recomputed = stamp;
  recomputed.policy = rsp::primal_policy::explicitly_recomputed;
  require(rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                      direct_plan, recomputed, direct_jvp, stream)
              == pg::program_status::launch_failed,
          "unsupported recomputed primal accepted");

  primitive unary_primitive(102, count, stream, state.value, forcing.value,
                            state.value, parameters.value, nn::local_operation::tanh);
  multiply_forward_state unary_descriptor{};
  pg::prepared_stage_v2 unary_stage{};
  vector unary_forward_output(count, stream, 0.f), unary_tangent_output(count, stream, 0.f);
  auto unary_bundle = forward_tanh_bundle(
      unary_descriptor, unary_stage, stage_id + 2, candidate_id, count);
  unary_bundle.bindings.front() = {
      state.value.data, unary_forward_output.value.data, nullptr};
  gh::dynamics::response_attachment unary_attachment{
      {stage_id + 2, candidate_id}, &unary_primitive.block, {}, {}, &activity_generation};
  gh::dynamics::direct_map_plan unary_plan{
      unary_bundle, gh::core::differentiated_object::vector_field, unary_attachment};
  const auto unary_stamp = rsp::retain_primal(unary_attachment);
  require(gh::dynamics::execute_direct_map(unary_plan, stream) == pg::program_status::success,
          "unary forward");
  require(std::abs(download(unary_forward_output, stream).front() - std::tanh(2.f)) < 2e-6f,
          "unary forward value");
  auto unary_jvp = unary_primitive.binding(
      nf::jvp, &state_direction.value, nullptr, nullptr,
      &unary_tangent_output.value, nullptr, nullptr);
  require(rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                      unary_plan, unary_stamp, unary_jvp, stream)
              == pg::program_status::success,
          "unary saved-primal response");

  // Feed sensitivities produced by the actual CE action into the local SVD.
  auto state_column = direct_primitive.binding(
      nf::jvp, &one_direction.value, &zero_direction.value, nullptr,
      &direct_output.value, nullptr, nullptr);
  require(rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                      direct_plan, stamp,
                                      state_column, stream) == pg::program_status::success,
          "state sensitivity");
  const float state_sensitivity = download(direct_output, stream).front();
  auto forcing_column = direct_primitive.binding(
      nf::jvp, &zero_direction.value, &one_direction.value, nullptr,
      &direct_output.value, nullptr, nullptr);
  require(rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                      direct_plan, stamp,
                                      forcing_column, stream) == pg::program_status::success,
          "forcing sensitivity");
  const float forcing_sensitivity = download(direct_output, stream).front();
  const std::array<double, 4> actual_sensitivity{
      state_sensitivity, forcing_sensitivity, state_sensitivity, forcing_sensitivity};
  const std::array<double, 2> scales{1., 1.};
  const auto local_null = gh::interrogation::rank_revealing_svd(
      actual_sensitivity, 2, 2, scales, 1e-10, "CE local add response");
  require(local_null.rank == 1 && local_null.right_nullspace.size() == 2 &&
          local_null.residual < 1e-10, "actual CE sensitivity nullspace");

  auto direct_vjp = direct_primitive.binding(
      nf::vjp, nullptr, nullptr, &cotangent.value, nullptr,
      &direct_left_adjoint.value, &direct_right_adjoint.value);
  require(rsp::execute_direct_map_vjp(gh::core::differentiated_object::vector_field,
                                      direct_plan, stamp, direct_vjp, stream)
              == pg::program_status::success,
          "direct map VJP");
  const auto direct_adj = download(direct_left_adjoint, stream);
  require(std::abs(count * .2f * .3f -
                   count * .2f * direct_adj.front()) < 2e-5f,
          "direct adjoint identity");
  ++activity_generation.value;
  require(rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                      direct_plan, stamp, direct_jvp, stream)
              == pg::program_status::launch_failed,
          "stale activity accepted");
  --activity_generation.value;
  ++state.value.generation.value;
  require(rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                      direct_plan, stamp, direct_jvp, stream)
              == pg::program_status::launch_failed,
          "stale state accepted");
  --state.value.generation.value;
  ++parameters.value.generation.value;
  require(rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                      direct_plan, stamp, direct_jvp, stream)
              == pg::program_status::launch_failed,
          "stale parameters accepted");
  --parameters.value.generation.value;
  ++forcing.value.generation.value;
  require(rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                      direct_plan, stamp, direct_jvp, stream)
              == pg::program_status::launch_failed,
          "stale forcing accepted");
  --forcing.value.generation.value;

  multiply_forward_state field_descriptor{};
  pg::prepared_stage_v2 field_stage{};
  auto field_bundle = forward_multiply_bundle(field_descriptor, field_stage, stage_id,
                                              candidate_id, count);
  const float h = .1f;
  const std::array<float, 4> stage_state_value{
      2.f, 2.f + .5f*h*1.f, 2.f + .5f*h*1.025f, 2.f + h*1.025625f};
  vector stage0(count, stream, stage_state_value[0], 7);
  vector stage1(count, stream, stage_state_value[1], 7);
  vector stage2(count, stream, stage_state_value[2], 7);
  vector stage3(count, stream, stage_state_value[3], 7);
  vector stage_forcing0(count, stream, .5f, 13);
  vector stage_forcing1(count, stream, .5f, 13);
  vector stage_forcing2(count, stream, .5f, 13);
  vector stage_forcing3(count, stream, .5f, 13);
  std::array<primitive, 4> fields{
      primitive(201, count, stream, stage0.value, stage_forcing0.value, state.value, parameters.value,
                nn::local_operation::multiply),
      primitive(202, count, stream, stage1.value, stage_forcing1.value, state.value, parameters.value,
                nn::local_operation::multiply),
      primitive(203, count, stream, stage2.value, stage_forcing2.value, state.value, parameters.value,
                nn::local_operation::multiply),
      primitive(204, count, stream, stage3.value, stage_forcing3.value, state.value, parameters.value,
                nn::local_operation::multiply)};
  gh::dynamics::response_attachment rk4_attachment{
      {stage_id, candidate_id}, &fields[0].block,
      {&fields[0].block, &fields[1].block, &fields[2].block, &fields[3].block},
      {&stage_forcing0.value, &stage_forcing1.value,
       &stage_forcing2.value, &stage_forcing3.value},
      &activity_generation};
  gh::dynamics::rk4_vector_field field{field_bundle, rk4_attachment};
  gh::dynamics::rk4_step_plan rk4(std::move(field), 800, candidate_id, count, h, stream);
  const auto rk4_stamp = rsp::retain_primal(rk4_attachment);

  vector trial_direction(count, stream, 0.f);
  vector dk1(count, stream, 0.f), dk2(count, stream, 0.f);
  vector dk3(count, stream, 0.f), dk4(count, stream, 0.f);
  vector tangent_packed(5*count, stream, 0.f), tangent_output(count, stream, 0.f);
  std::array<df::response_binding<float>, 4> jvp_bindings{
      fields[0].binding(nf::jvp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
      fields[1].binding(nf::jvp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
      fields[2].binding(nf::jvp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
      fields[3].binding(nf::jvp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)};
  rsp::rk4_jvp_buffers jvp_buffers{
      state_direction.value,
      {&forcing_direction.value, &forcing_direction.value,
       &forcing_direction.value, &forcing_direction.value},
      trial_direction.value, {&dk1.value, &dk2.value, &dk3.value, &dk4.value},
      tangent_packed.value, tangent_output.value};
  require(rsp::execute_rk4_jvp(rk4, rk4_stamp, jvp_bindings, jvp_buffers, stream)
              == pg::program_status::success,
          "RK4 JVP");
  const auto tangent = download(tangent_output, stream);

  const float epsilon = 1e-3f;
  vector plus(count, stream, 2.f + epsilon*.2f, 7);
  vector minus(count, stream, 2.f - epsilon*.2f, 7);
  vector plus_forcing(count, stream, .5f + epsilon*.1f, 11);
  vector minus_forcing(count, stream, .5f - epsilon*.1f, 11);
  vector trial(count, stream, 0.f), k1(count, stream, 0.f), k2(count, stream, 0.f);
  vector k3(count, stream, 0.f), k4(count, stream, 0.f), packed(5*count, stream, 0.f);
  vector forward_output(count, stream, 0.f);
  gh::dynamics::rk4_resident_scratch forward_scratch{
      trial.value, k1.value, k2.value, k3.value, k4.value, packed.value};
  const auto plus_stages = gh::dynamics::make_rk4_stage_bindings(0., h, plus_forcing.value);
  const auto minus_stages = gh::dynamics::make_rk4_stage_bindings(0., h, minus_forcing.value);
  require(gh::dynamics::execute_rk4_step(rk4, plus.value, plus_stages, forward_scratch,
                                        forward_output.value, stream)
              == pg::program_status::success,
          "plus RK4");
  const auto plus_result = download(forward_output, stream);
  require(gh::dynamics::execute_rk4_step(rk4, minus.value, minus_stages, forward_scratch,
                                        forward_output.value, stream)
              == pg::program_status::success,
          "minus RK4");
  const auto minus_result = download(forward_output, stream);
  const float finite_difference = (plus_result.front() - minus_result.front())/(2.f*epsilon);
  require(std::abs(tangent.front() - finite_difference) < 3e-4f,
          "RK4 finite-difference mismatch");

  vector adjoint_state(count, stream, 0.f), adjoint_temp(count, stream, 0.f);
  vector stage_adjoint(count, stream, 0.f);
  vector ak1(count, stream, 0.f), ak2(count, stream, 0.f);
  vector ak3(count, stream, 0.f), ak4(count, stream, 0.f);
  vector au1(count, stream, 0.f), au2(count, stream, 0.f);
  vector au3(count, stream, 0.f), au4(count, stream, 0.f);
  std::array<df::response_binding<float>, 4> vjp_bindings{
      fields[0].binding(nf::vjp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
      fields[1].binding(nf::vjp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
      fields[2].binding(nf::vjp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
      fields[3].binding(nf::vjp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)};
  rsp::rk4_vjp_buffers vjp_buffers{
      cotangent.value, zero_direction.value, adjoint_state.value, adjoint_temp.value,
      stage_adjoint.value, {&ak1.value, &ak2.value, &ak3.value, &ak4.value},
      {&au1.value, &au2.value, &au3.value, &au4.value}};
  require(rsp::execute_rk4_vjp(rk4, rk4_stamp, vjp_bindings, vjp_buffers, stream)
              == pg::program_status::success,
          "RK4 VJP");
  const auto state_adjoint = download(adjoint_state, stream);
  const float forcing_adjoint = download(au1, stream).front() + download(au2, stream).front()
      + download(au3, stream).front() + download(au4, stream).front();
  require(std::abs(count*tangent.front()*.3f -
                   count*(.2f*state_adjoint.front() + .1f*forcing_adjoint)) < 3e-4f,
          "RK4 adjoint identity");

  ++stage_forcing3.value.generation.value;
  require(rsp::execute_rk4_jvp(rk4, rk4_stamp, jvp_bindings, jvp_buffers, stream)
              == pg::program_status::launch_failed,
          "stale stage-3 forcing accepted");
  --stage_forcing3.value.generation.value;
  auto decoy_attachment = rk4_attachment;
  decoy_attachment.forcing_owners[3] = &forcing.value;
  require(rsp::retain_primal(decoy_attachment).state_owner == nullptr,
          "decoy stage-3 forcing owner retained");

  gh::dynamics::direct_map_plan observation_plan{
      direct_bundle, gh::core::differentiated_object::observation, direct_attachment};
  auto observation_jvp = direct_primitive.binding(
      nf::jvp, &state_direction.value, &forcing_direction.value, nullptr,
      &direct_output.value, nullptr, nullptr);
  require(rsp::execute_observation_jvp(observation_plan, stamp,
                                      observation_jvp, stream)
              == pg::program_status::success,
          "observation response");
  const auto observation_tangent = download(direct_output, stream).front();
  require(rsp::execute_observation_vjp(observation_plan, stamp,
                                      direct_vjp, stream)
              == pg::program_status::success,
          "observation adjoint response");
  auto observation_plus = observation_plan;
  observation_plus.stage.bindings.front() = {
      plus.value.data, direct_output.value.data, plus_forcing.value.data};
  require(gh::dynamics::execute_direct_map(observation_plus, stream)
              == pg::program_status::success,
          "observation plus forward");
  const auto observation_plus_value = download(direct_output, stream).front();
  auto observation_minus = observation_plan;
  observation_minus.stage.bindings.front() = {
      minus.value.data, direct_output.value.data, minus_forcing.value.data};
  require(gh::dynamics::execute_direct_map(observation_minus, stream)
              == pg::program_status::success,
          "observation minus forward");
  const auto observation_minus_value = download(direct_output, stream).front();
  const auto observation_fd = (observation_plus_value-observation_minus_value)/(2.f*epsilon);
  require(std::abs(observation_fd-observation_tangent) < 2e-4f,
          "observation finite difference");
  const auto observation_left = download(direct_left_adjoint, stream).front();
  const auto observation_right = download(direct_right_adjoint, stream).front();
  require(std::abs(.3f*.3f-(.2f*observation_left+.1f*observation_right)) < 2e-5f,
          "observation adjoint identity");
  require(rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                      observation_plan, stamp, direct_jvp, stream)
              == pg::program_status::launch_failed,
          "observation relabeled as vector field");

  vector rollout_primal(count, stream, 2.5f, 7);
  primitive rollout_second_primitive(101, count, stream, rollout_primal.value,
                                     forcing.value, rollout_primal.value, parameters.value);
  nn::linear_stage rollout_second_descriptor{};
  pg::prepared_stage_v2 rollout_second_stage{};
  auto rollout_second_bundle = forward_add_bundle(
      rollout_second_descriptor, rollout_second_stage, stage_id + 1, candidate_id, count);
  gh::dynamics::response_attachment rollout_second_attachment{
      {stage_id + 1, candidate_id}, &rollout_second_primitive.block, {}, {&forcing.value},
      &activity_generation};
  std::array<gh::dynamics::direct_map_plan, 2> rollout{
      gh::dynamics::direct_map_plan{direct_bundle,
          gh::core::differentiated_object::implemented_rollout, direct_attachment},
      gh::dynamics::direct_map_plan{rollout_second_bundle,
          gh::core::differentiated_object::implemented_rollout, rollout_second_attachment}};
  vector rollout_mid(count, stream, 0.f), rollout_out(count, stream, 0.f);
  rollout[0].stage.bindings.front() = {
      state.value.data, rollout_mid.value.data, forcing.value.data};
  rollout[1].stage.bindings.front() = {
      rollout_mid.value.data, rollout_out.value.data, forcing.value.data};
  require(gh::dynamics::execute_direct_map(rollout[0], stream)
              == pg::program_status::success &&
          gh::dynamics::execute_direct_map(rollout[1], stream)
              == pg::program_status::success,
          "ordered rollout forward");
  for (float value : download(rollout_out, stream))
    require(std::abs(value - 3.f) < 2e-6f, "ordered rollout value");
  std::array<df::response_binding<float>, 2> rollout_bindings{
      direct_primitive.binding(nf::jvp, &state_direction.value, &zero_direction.value,
                               nullptr, &rollout_mid.value, nullptr, nullptr),
      rollout_second_primitive.binding(nf::jvp, &rollout_mid.value, &zero_direction.value,
                                       nullptr, &rollout_out.value, nullptr, nullptr)};
  std::array<rsp::saved_primal, 2> rollout_stamps{
      stamp, rsp::retain_primal(rollout_second_attachment)};
  require(rsp::execute_rollout_jvp(rollout, rollout_stamps, rollout_bindings, stream)
              == pg::program_status::success,
          "rollout response");
  for (float value : download(rollout_out, stream))
    require(std::abs(value - .2f) < 2e-6f, "rollout analytic response");

  vector rollout_state0(count, stream, 0.f), rollout_state1(count, stream, 0.f);
  vector rollout_aux0(count, stream, 0.f), rollout_aux1(count, stream, 0.f);
  std::array<df::response_binding<float>, 2> rollout_vjp_bindings{
      direct_primitive.binding(nf::vjp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
      rollout_second_primitive.binding(nf::vjp, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)};
  std::array<nn::resident_vector*, 2> rollout_state_adjoints{
      &rollout_state0.value, &rollout_state1.value};
  std::array<nn::resident_vector*, 2> rollout_aux_adjoints{
      &rollout_aux0.value, &rollout_aux1.value};
  require(rsp::execute_rollout_vjp(
              rollout, rollout_stamps, rollout_vjp_bindings,
              {cotangent.value, rollout_state_adjoints, rollout_aux_adjoints}, stream)
              == pg::program_status::success,
          "rollout adjoint response");
  const auto rollout_input_adjoint = download(rollout_state0, stream);
  require(std::abs(count*.2f*.3f - count*.2f*rollout_input_adjoint.front()) < 2e-5f,
          "rollout adjoint identity");

  cuda_require(cudaStreamDestroy(stream), "destroy stream");
}
