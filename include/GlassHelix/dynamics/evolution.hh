#pragma once
#include <GlassHelix/core/request.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>
#include <Cellerator/compute/operation/native_numeric/device_linear.hh>
#include <Cellerator/execution/program/program_v2.h>
#include <algorithm>
#include <array>
#include <deque>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace glasshelix::dynamics {
namespace program = cellerator::execution::program;
namespace operation = cellerator::compute::operation::v2;

// The Cellerator stage factories own all arithmetic, resident buffers and
// readiness. GlassHelix retains only the scientific ordering of those stages.
struct prepared_stage_bundle {
  program::prepared_program_v2 program{};
  std::vector<program::launch_binding_v2> bindings;
};

struct resident_instance {
  operation::stable_id identity{};
  cellerator::execution::value_generation state_generation{};
  cellerator::execution::value_generation parameter_generation{};
  std::string representation;
};

struct primal_checkpoint {
  operation::stable_id instance{};
  cellerator::execution::value_generation state_generation{};
  cellerator::execution::value_generation parameter_generation{};
  double time = 0;
  std::size_t completed_steps = 0;
  // CE allocates and copies this owned resident snapshot; no ping-pong state
  // pointer is treated as a retained primal.
  std::shared_ptr<cellerator::compute::native_numeric::resident_vector> retained_primal;
};

inline bool valid_checkpoint(const primal_checkpoint& checkpoint,
                             const resident_instance& instance) noexcept {
  return operation::same_stable_id(checkpoint.instance, instance.identity)
    && checkpoint.state_generation.value == instance.state_generation.value
    && checkpoint.parameter_generation.value == instance.parameter_generation.value
    && checkpoint.retained_primal && checkpoint.retained_primal->data;
}

class bounded_primal_history {
  std::size_t capacity_;
  std::deque<primal_checkpoint> checkpoints_;
public:
  explicit bounded_primal_history(std::size_t capacity) : capacity_(capacity) {
    if (!capacity_) throw std::invalid_argument("checkpoint capacity required");
  }
  void record(primal_checkpoint checkpoint) {
    if (!std::isfinite(checkpoint.time) || !checkpoint.retained_primal || !checkpoint.retained_primal->data)
      throw std::invalid_argument("invalid primal checkpoint");
    if (checkpoints_.size() == capacity_) checkpoints_.pop_front();
    checkpoints_.push_back(checkpoint);
  }
  const primal_checkpoint& require_current(const resident_instance& instance,
                                           std::size_t completed_steps) const {
    for (auto it = checkpoints_.rbegin(); it != checkpoints_.rend(); ++it)
      if (it->completed_steps == completed_steps && valid_checkpoint(*it, instance)) return *it;
    throw std::invalid_argument("stale or unavailable primal checkpoint");
  }
  std::size_t size() const noexcept { return checkpoints_.size(); }
};

struct forcing_interval { double begin, end; std::size_t knot_index; };

struct rk4_interval {
  forcing_interval forcing;
  std::size_t steps = 0;
};

inline std::vector<forcing_interval> split_forcing_intervals(
    const core::rk4_request& request) {
  if (!core::valid(request)) throw std::invalid_argument("invalid RK4 request");
  std::vector<double> boundaries{request.begin};
  for (const auto& knot : request.forcing.knots)
    if (knot.time > request.begin && knot.time < request.end) boundaries.push_back(knot.time);
  boundaries.push_back(request.end);
  std::vector<forcing_interval> result;
  for (std::size_t i = 0; i + 1 < boundaries.size(); ++i) {
    const auto position = std::upper_bound(request.forcing.knots.begin(), request.forcing.knots.end(),
      boundaries[i], [](double time, const core::forcing_knot& knot) { return time < knot.time; });
    // The interval ending at a declared discontinuity keeps the preceding
    // forcing value through its endpoint RK4 stage. The next interval begins
    // with the new value.
    const auto index = position == request.forcing.knots.begin() ? 0 : static_cast<std::size_t>(position - request.forcing.knots.begin() - 1);
    result.push_back({boundaries[i], boundaries[i + 1], index});
  }
  return result;
}

// A supplied discontinuity is part of the executed discrete calculation.  It
// must therefore lie on a fixed-step boundary: silently taking an RK stage on
// both sides of a knot would describe neither declared forcing convention.
inline std::vector<rk4_interval> make_rk4_schedule(
    const core::rk4_request& request) {
  const auto intervals = split_forcing_intervals(request);
  std::vector<rk4_interval> schedule;
  schedule.reserve(intervals.size());
  for (const auto& interval : intervals) {
    const double raw_steps = (interval.end - interval.begin) / request.dt;
    const auto steps = static_cast<std::size_t>(std::llround(raw_steps));
    const double tolerance = 32.0 * std::numeric_limits<double>::epsilon()
      * std::max({1.0, std::abs(interval.begin), std::abs(interval.end), std::abs(request.dt)});
    if (!steps || std::abs(raw_steps - static_cast<double>(steps)) > tolerance)
      throw std::invalid_argument("RK4 forcing discontinuity is not on a fixed-step boundary");
    schedule.push_back({interval, steps});
  }
  return schedule;
}

inline program::program_status submit(const prepared_stage_bundle& bundle,
                                      void* caller_stream) noexcept {
  return program::execute_prepared_program_v2(bundle.program, bundle.bindings.data(),
                                               bundle.bindings.size(), caller_stream);
}

struct direct_map_plan { prepared_stage_bundle stage; };

// The supplied CE program is a vector field with one binding. GlassHelix only
// patches the state (`input`), forcing (`values`), and derivative (`output`)
// slots for each mathematical RK4 stage; all other CE binding fields remain
// provider-owned.
struct rk4_vector_field {
  prepared_stage_bundle program;
};

struct rk4_stage_binding {
  double time = 0;
  const cellerator::compute::native_numeric::resident_vector* forcing = nullptr;
};

inline std::array<rk4_stage_binding, 4> make_rk4_stage_bindings(
    double begin, double step,
    const std::array<const cellerator::compute::native_numeric::resident_vector*, 4>& forcing) {
  return {{{begin, forcing[0]}, {begin + .5 * step, forcing[1]},
           {begin + .5 * step, forcing[2]}, {begin + step, forcing[3]}}};
}
inline std::array<rk4_stage_binding, 4> make_rk4_stage_bindings(
    double begin, double step,
    const cellerator::compute::native_numeric::resident_vector& forcing) {
  return make_rk4_stage_bindings(begin, step, {{&forcing, &forcing, &forcing, &forcing}});
}

// Every vector is CE resident. `packed` has five contiguous vector-width
// regions, used solely as the documented CE weighted_sum4 workspace.
struct rk4_resident_scratch {
  cellerator::compute::native_numeric::resident_vector& trial;
  cellerator::compute::native_numeric::resident_vector& k1;
  cellerator::compute::native_numeric::resident_vector& k2;
  cellerator::compute::native_numeric::resident_vector& k3;
  cellerator::compute::native_numeric::resident_vector& k4;
  cellerator::compute::native_numeric::resident_vector& packed;
};

// The final RK4 combination is a CE-owned resident-vector stage.  Its backing
// descriptor stays with this object because program_v2 intentionally borrows
// prepared state.  `workspace` must contain {base,k1,k2,k3,k4}, each with the
// same number of FP32 elements; CE validates that contract before launch.
struct rk4_combine_stage {
  cellerator::compute::native_numeric::linear_stage descriptor{
    cellerator::compute::native_numeric::linear_kind::weighted_sum4, 0,
    cellerator::compute::native_numeric::device_representation::f32, 0.0f, 0.0f};
  program::prepared_stage_v2 stage{};
  rk4_combine_stage(std::uint64_t stage_id, std::uint64_t candidate_id,
                    std::uint64_t elements, float step_size)
      : descriptor{cellerator::compute::native_numeric::linear_kind::weighted_sum4,
                   elements,
                   cellerator::compute::native_numeric::device_representation::f32,
                   step_size, 0.0f},
        stage(cellerator::compute::native_numeric::make_linear_stage(
            stage_id, candidate_id, &descriptor)) {}
};


class rk4_step_plan {
  using linear_stage = cellerator::compute::native_numeric::linear_stage;
  using linear_kind = cellerator::compute::native_numeric::linear_kind;
  using device_representation = cellerator::compute::native_numeric::device_representation;
  linear_stage copy_descriptor_{linear_kind::copy, 0, device_representation::f32, 1.f, 0.f};
  linear_stage half_accumulate_descriptor_{linear_kind::axpby, 0, device_representation::f32, 1.f, 0.f};
  linear_stage full_accumulate_descriptor_{linear_kind::axpby, 0, device_representation::f32, 1.f, 0.f};
  rk4_combine_stage combine_;
  std::uint64_t elements_ = 0;
  float step_ = 0.f;
  void* stream_ = nullptr;
  program::prepared_stage_v2 copy_stage_{};
  program::prepared_stage_v2 half_accumulate_stage_{};
  program::prepared_stage_v2 full_accumulate_stage_{};
  rk4_vector_field field_;
public:
  rk4_step_plan(rk4_vector_field field, std::uint64_t stage_id,
                std::uint64_t candidate_id, std::uint64_t elements, float step,
                void* stream = nullptr)
      : copy_descriptor_{linear_kind::copy, elements, device_representation::f32, 1.f, 0.f},
        half_accumulate_descriptor_{linear_kind::axpby, elements, device_representation::f32, 1.f, .5f * step},
        full_accumulate_descriptor_{linear_kind::axpby, elements, device_representation::f32, 1.f, step},
        combine_(stage_id + 3, candidate_id, elements, step), elements_(elements), step_(step), stream_(stream), field_(std::move(field)) {
    copy_stage_ = cellerator::compute::native_numeric::make_linear_stage(stage_id, candidate_id, &copy_descriptor_);
    half_accumulate_stage_ = cellerator::compute::native_numeric::make_linear_stage(stage_id + 1, candidate_id, &half_accumulate_descriptor_);
    full_accumulate_stage_ = cellerator::compute::native_numeric::make_linear_stage(stage_id + 2, candidate_id, &full_accumulate_descriptor_);
  }
  rk4_step_plan(const rk4_step_plan&) = delete;
  rk4_step_plan& operator=(const rk4_step_plan&) = delete;
  const rk4_vector_field& field() const noexcept { return field_; }
  const program::prepared_stage_v2& copy_stage() const noexcept { return copy_stage_; }
  const program::prepared_stage_v2& half_accumulate_stage() const noexcept { return half_accumulate_stage_; }
  const program::prepared_stage_v2& full_accumulate_stage() const noexcept { return full_accumulate_stage_; }
  const rk4_combine_stage& combine() const noexcept { return combine_; }
  std::uint64_t elements() const noexcept { return elements_; }
  float step() const noexcept { return step_; }
  void* stream() const noexcept { return stream_; }
};

inline program::program_status execute_direct_map(const direct_map_plan& plan,
                                                   void* caller_stream) noexcept {
  return submit(plan.stage, caller_stream);
}

// The two bindings are caller-owned CE resident buffers.  This routine only
// schedules the prepared Cellerator program and swaps those buffers; it never
// downloads state or supplies a GlassHelix numerical fallback.
inline program::program_status execute_resident_direct_rollout(
    const direct_map_plan& plan, program::launch_binding_v2 first,
    program::launch_binding_v2 second, std::size_t steps,
    void* caller_stream) noexcept {
  if (!steps || !first.input || !first.output || !second.input || !second.output
      || first.input != second.output || first.output != second.input)
    return program::program_status::invalid_argument;
  for (std::size_t step = 0; step < steps; ++step) {
    auto bundle = plan.stage;
    bundle.bindings.assign(1, step % 2 == 0 ? first : second);
    const auto status = submit(bundle, caller_stream);
    if (status != program::program_status::success) return status;
  }
  return program::program_status::success;
}

inline program::program_status submit_one(const program::prepared_stage_v2& stage,
                                          program::launch_binding_v2 binding,
                                          void* caller_stream) noexcept {
  const program::prepared_program_v2 program{2, 0, &stage, 1, nullptr, 0};
  return program::execute_prepared_program_v2(program, &binding, 1, caller_stream);
}

// Retention is an explicit CE resident copy. The caller supplies the same
// stream bound to the plan, so uploads/readiness on that stream are a caller
// prerequisite; this API never invents cross-stream event synchronization.
inline program::program_status capture_primal_checkpoint(
    primal_checkpoint* checkpoint,
    const cellerator::compute::native_numeric::resident_vector& source,
    void* caller_stream) noexcept {
  using namespace cellerator::compute::native_numeric;
  if (!checkpoint || !source.data || !source.elements
      || source.representation != device_representation::f32) return program::program_status::invalid_argument;
  auto retained = std::shared_ptr<resident_vector>(new resident_vector{}, [](resident_vector* value) {
    (void)release(value); delete value;
  });
  if (allocate(retained.get(), source.elements, source.representation, source.device_ordinal) != cudaSuccess)
    return program::program_status::launch_failed;
  linear_stage descriptor{linear_kind::copy, source.elements, source.representation, 1.f, 0.f};
  const auto stage = make_linear_stage(9001, 9001, &descriptor);
  if (const auto status = submit_one(stage, {source.data, retained->data}, caller_stream);
      status != program::program_status::success) return status;
  checkpoint->retained_primal = std::move(retained);
  return program::program_status::success;
}

inline program::program_status execute_rk4_step(
    const rk4_step_plan& plan,
    const cellerator::compute::native_numeric::resident_vector& state,
    const std::array<rk4_stage_binding, 4>& stages,
    rk4_resident_scratch scratch,
    cellerator::compute::native_numeric::resident_vector& output,
    void* caller_stream) noexcept {
  using cellerator::compute::native_numeric::device_representation;
  const auto valid = [&state](const auto& value) {
    return value.data && value.elements == state.elements
      && value.representation == device_representation::f32
      && value.device_ordinal == state.device_ordinal;
  };
  const std::array<const void*, 8> addresses{state.data, output.data, scratch.trial.data,
      scratch.k1.data, scratch.k2.data, scratch.k3.data, scratch.k4.data, scratch.packed.data};
  const auto aliases = [&addresses] {
    for (std::size_t i = 0; i < addresses.size(); ++i)
      for (std::size_t j = i + 1; j < addresses.size(); ++j)
        if (addresses[i] == addresses[j]) return true;
    return false;
  };
  if (!valid(state) || !valid(output) || !valid(scratch.trial) || !valid(scratch.k1)
      || !valid(scratch.k2) || !valid(scratch.k3) || !valid(scratch.k4)
      || !state.elements || state.elements > std::numeric_limits<std::uint64_t>::max() / 5
      || state.elements != plan.elements() || !std::isfinite(plan.step()) || plan.step() <= 0.f
      || caller_stream != plan.stream() || output.generation.value == std::numeric_limits<std::uint64_t>::max()
      || !scratch.packed.data || scratch.packed.elements != 5 * state.elements
      || scratch.packed.representation != device_representation::f32
      || scratch.packed.device_ordinal != state.device_ordinal || aliases()
      || plan.field().program.bindings.size() != 1
      || !std::isfinite(stages[0].time)
      || std::abs((stages[1].time - stages[0].time) - .5 * plan.step()) > 1e-6 * plan.step()
      || std::abs((stages[2].time - stages[1].time)) > 1e-6 * plan.step()
      || std::abs((stages[3].time - stages[0].time) - plan.step()) > 1e-6 * plan.step())
    return program::program_status::invalid_argument;
  auto vector_field = [&](const auto& input, const rk4_stage_binding& stage, auto& derivative) {
    // The declared field contract is FP32 state-width forcing on this device.
    if (!stage.forcing || !valid(*stage.forcing)
        || stage.forcing->data == output.data || stage.forcing->data == scratch.trial.data
        || stage.forcing->data == scratch.k1.data || stage.forcing->data == scratch.k2.data
        || stage.forcing->data == scratch.k3.data || stage.forcing->data == scratch.k4.data
        || stage.forcing->data == scratch.packed.data) return program::program_status::invalid_argument;
    auto binding = plan.field().program.bindings.front();
    binding.input = input.data;
    binding.values = stage.forcing->data;
    binding.output = derivative.data;
    return program::execute_prepared_program_v2(plan.field().program.program, &binding, 1, caller_stream);
  };
  auto accumulate = [&](const auto& stage, const auto& derivative) {
    return submit_one(stage, {state.data, scratch.trial.data, derivative.data}, caller_stream);
  };
  if (auto status = vector_field(state, stages[0], scratch.k1); status != program::program_status::success) return status;
  if (auto status = accumulate(plan.half_accumulate_stage(), scratch.k1); status != program::program_status::success) return status;
  if (auto status = vector_field(scratch.trial, stages[1], scratch.k2); status != program::program_status::success) return status;
  if (auto status = accumulate(plan.half_accumulate_stage(), scratch.k2); status != program::program_status::success) return status;
  if (auto status = vector_field(scratch.trial, stages[2], scratch.k3); status != program::program_status::success) return status;
  if (auto status = accumulate(plan.full_accumulate_stage(), scratch.k3); status != program::program_status::success) return status;
  if (auto status = vector_field(scratch.trial, stages[3], scratch.k4); status != program::program_status::success) return status;

  auto* packed = static_cast<std::byte*>(scratch.packed.data);
  const auto stride = state.elements * sizeof(float);
  const std::array<const cellerator::compute::native_numeric::resident_vector*, 5> values{
      &state, &scratch.k1, &scratch.k2, &scratch.k3, &scratch.k4};
  for (std::size_t i = 0; i < values.size(); ++i)
    if (auto status = submit_one(plan.copy_stage(), {values[i]->data, packed + i * stride}, caller_stream);
        status != program::program_status::success) return status;
  const auto status = submit_one(plan.combine().stage,
      {scratch.packed.data, output.data, nullptr, scratch.packed.data, 5 * stride}, caller_stream);
  if (status == program::program_status::success) ++output.generation.value;
  return status;
}
} // namespace glasshelix::dynamics
