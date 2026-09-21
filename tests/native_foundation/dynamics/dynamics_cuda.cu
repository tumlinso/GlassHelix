#include <GlassHelix/dynamics/evolution.hh>

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace ce = cellerator::compute::native_numeric;
namespace program = cellerator::execution::program;
namespace gh = glasshelix::dynamics;

namespace {
constexpr std::size_t width = 33;

void check(cudaError_t status, const char* message) {
  if (status != cudaSuccess) throw std::runtime_error(message);
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct device_vector {
  ce::resident_vector value{};
  explicit device_vector(std::size_t elements) {
    check(ce::allocate(&value, elements, ce::device_representation::f32, 0), "device allocation failed");
  }
  ~device_vector() { (void)ce::release(&value); }
  device_vector(const device_vector&) = delete;
};
void upload(device_vector& target, const std::array<float, width>& values, std::uint64_t generation = 1) {
  check(ce::upload(target.value, values.data(), values.size(), {generation}, nullptr), "device upload failed");
}
std::array<float, width> download(const device_vector& source) {
  std::array<float, width> values{};
  check(ce::download(source.value, values.data(), values.size(), nullptr), "device download failed");
  check(cudaDeviceSynchronize(), "device synchronization failed");
  return values;
}
struct rk4_vectors {
  device_vector trial{width}, k1{width}, k2{width}, k3{width}, k4{width}, packed{5 * width};
  gh::rk4_resident_scratch scratch() { return {trial.value, k1.value, k2.value, k3.value, k4.value, packed.value}; }
};

// Test-only CPU oracle. Production arithmetic remains entirely in CE stages.
float cpu_rk4_affine(float y, float forcing, float h) {
  const auto f = [forcing](float x) { return x + forcing; };
  const auto k1 = f(y);
  const auto k2 = f(y + .5f * h * k1);
  const auto k3 = f(y + .5f * h * k2);
  const auto k4 = f(y + h * k3);
  return y + h * (k1 + 2.f * k2 + 2.f * k3 + k4) / 6.f;
}

void rk4_steps(const gh::rk4_step_plan& plan, device_vector& state, device_vector& next,
               const device_vector& forcing, rk4_vectors& vectors, float begin,
               float h, std::size_t count) {
  for (std::size_t step = 0; step < count; ++step) {
    const auto stages = gh::make_rk4_stage_bindings(begin + h * static_cast<float>(step), h, forcing.value);
    const auto time = begin + h * static_cast<float>(step);
    require(std::abs(stages[0].time - time) < 1e-6 && std::abs(stages[1].time - (time + .5f * h)) < 1e-6
            && std::abs(stages[2].time - stages[1].time) < 1e-6
            && std::abs(stages[3].time - (time + h)) < 1e-6, "incorrect RK4 stage times");
    require(gh::execute_rk4_step(plan, state.value, stages, vectors.scratch(), next.value, nullptr)
                == program::program_status::success, "four-stage CE RK4 launch failed");
    std::swap(state.value.data, next.value.data);
  }
}
} // namespace

int main() try {
  std::array<float, width> base{}, forcing_values{}, forcing_zero{};
  for (std::size_t i = 0; i < width; ++i) {
    base[i] = .01f * static_cast<float>(i);
    forcing_values[i] = .25f;
  }

  // Supplied CE vector field f(y,u)=y+u. GH may only patch its three explicit slots.
  ce::linear_stage field_descriptor{ce::linear_kind::axpby, width, ce::device_representation::f32, 1.f, 1.f};
  const auto field_stage = ce::make_linear_stage(40, 9, &field_descriptor);
  program::prepared_program_v2 field_program{2, 0, &field_stage, 1, nullptr, 0};
  gh::rk4_vector_field field{{field_program, {program::launch_binding_v2{nullptr, nullptr, nullptr, nullptr, 0}}}};
  constexpr float h = .05f;
  gh::rk4_step_plan plan(std::move(field), 50, 9, width, h);

  device_vector current(width), next(width), forcing(width), zero(width);
  rk4_vectors vectors;
  upload(current, base); upload(forcing, forcing_values); upload(zero, forcing_zero);
  rk4_steps(plan, current, next, forcing, vectors, 0.f, h, 20);
  const auto result = download(current);
  for (std::size_t i = 0; i < width; ++i) {
    float expected = base[i];
    for (int step = 0; step < 20; ++step) expected = cpu_rk4_affine(expected, forcing_values[i], h);
    require(std::abs(result[i] - expected) < 2e-5f, "CE RK4 did not execute all vector-field stages");
  }

  // Refinement is measured against the analytic y'=y solution, independent of CE.
  device_vector coarse(width), coarse_next(width), fine(width), fine_next(width);
  rk4_vectors coarse_vectors, fine_vectors;
  upload(coarse, base); upload(fine, base);
  gh::rk4_step_plan coarse_plan({{field_program, {program::launch_binding_v2{}}}}, 60, 9, width, .2f);
  gh::rk4_step_plan fine_plan({{field_program, {program::launch_binding_v2{}}}}, 70, 9, width, .1f);
  rk4_steps(coarse_plan, coarse, coarse_next, zero, coarse_vectors, 0.f, .2f, 5);
  rk4_steps(fine_plan, fine, fine_next, zero, fine_vectors, 0.f, .1f, 10);
  const auto coarse_result = download(coarse), fine_result = download(fine);
  for (std::size_t i = 1; i < width; ++i) {
    const auto exact = base[i] * std::exp(1.f);
    require(std::abs(fine_result[i] - exact) < std::abs(coarse_result[i] - exact), "RK4 refinement did not converge");
  }

  // A declared forcing discontinuity is on a step boundary: all four stage
  // bindings for each step use that interval's supplied CE resident forcing.
  device_vector discontinuous(width), discontinuous_next(width);
  rk4_vectors discontinuous_vectors;
  std::array<float, width> zero_base{};
  upload(discontinuous, zero_base);
  gh::rk4_step_plan forcing_plan({{field_program, {program::launch_binding_v2{}}}}, 80, 9, width, .25f);
  rk4_steps(forcing_plan, discontinuous, discontinuous_next, forcing, discontinuous_vectors, 0.f, .25f, 3);
  rk4_steps(forcing_plan, discontinuous, discontinuous_next, zero, discontinuous_vectors, .75f, .25f, 1);
  const auto discontinuous_result = download(discontinuous);
  const auto expected_discontinuous = [&] {
    float y = 0; for (int i = 0; i < 3; ++i) y = cpu_rk4_affine(y, .25f, .25f);
    return cpu_rk4_affine(y, 0.f, .25f);
  }();
  require(std::abs(discontinuous_result[0] - expected_discontinuous) < 2e-5f,
          "forcing discontinuity crossed an RK4 stage");

  // Independent resident trajectories and a retained real CE primal checkpoint.
  device_vector other(width), other_next(width); rk4_vectors other_vectors;
  std::array<float, width> other_base{}; other_base.fill(2.f); upload(other, other_base);
  rk4_steps(plan, other, other_next, forcing, other_vectors, 0.f, h, 1);
  require(std::abs(download(other)[0] - download(current)[0]) > .1f, "resident trajectories were coupled");
  gh::resident_instance instance{{1, 0}, current.value.generation, {7}, "f32"};
  gh::bounded_primal_history history(1);
  history.record({instance.identity, instance.state_generation, instance.parameter_generation,
                  1., 20, current.value.data});
  (void)history.require_current(instance, 20);
  ++instance.parameter_generation.value;
  bool stale = false; try { (void)history.require_current(instance, 20); } catch (const std::invalid_argument&) { stale = true; }
  require(stale, "stale parameter checkpoint was accepted");
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
