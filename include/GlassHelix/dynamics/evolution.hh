#pragma once
#include <GlassHelix/core/request.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>
#include <Cellerator/compute/operation/native_numeric/device_linear.hh>
#include <Cellerator/execution/program/program_v2.h>
#include <algorithm>
#include <array>
#include <deque>
#include <limits>
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
  // A CE-owned retained state handle is intentionally opaque to GlassHelix.
  const void* retained_primal = nullptr;
};

inline bool valid_checkpoint(const primal_checkpoint& checkpoint,
                             const resident_instance& instance) noexcept {
  return operation::same_stable_id(checkpoint.instance, instance.identity)
    && checkpoint.state_generation.value == instance.state_generation.value
    && checkpoint.parameter_generation.value == instance.parameter_generation.value
    && checkpoint.retained_primal != nullptr;
}

class bounded_primal_history {
  std::size_t capacity_;
  std::deque<primal_checkpoint> checkpoints_;
public:
  explicit bounded_primal_history(std::size_t capacity) : capacity_(capacity) {
    if (!capacity_) throw std::invalid_argument("checkpoint capacity required");
  }
  void record(primal_checkpoint checkpoint) {
    if (!std::isfinite(checkpoint.time) || !checkpoint.retained_primal)
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
struct rk4_plan { std::array<prepared_stage_bundle, 5> stages; };

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

inline program::program_status execute_rk4_step(const rk4_plan& plan,
                                                 void* caller_stream) noexcept {
  for (const auto& stage : plan.stages)
    if (const auto status = submit(stage, caller_stream); status != program::program_status::success)
      return status;
  return program::program_status::success;
}
} // namespace glasshelix::dynamics
