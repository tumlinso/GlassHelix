#include <GlassHelix/dynamics/evolution.hh>

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <stdexcept>

namespace ce = cellerator::compute::native_numeric;
namespace program = cellerator::execution::program;

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
} // namespace

int main() try {
  std::array<float, width> base{}, k1{}, k2{}, k3{}, k4{}, forcing{};
  for (std::size_t i = 0; i < width; ++i) {
    base[i] = .01f * static_cast<float>(i);
    k1[i] = 1.0f; k2[i] = 2.0f; k3[i] = 3.0f; k4[i] = 4.0f;
    forcing[i] = .25f;
  }
  device_vector workspace(5 * width), combined(width), current(width), next(width), input(width);
  std::array<float, 5 * width> packed{};
  for (std::size_t i = 0; i < width; ++i) {
    packed[i] = base[i]; packed[width + i] = k1[i]; packed[2 * width + i] = k2[i];
    packed[3 * width + i] = k3[i]; packed[4 * width + i] = k4[i];
  }
  check(ce::upload(workspace.value, packed.data(), packed.size(), {1}, nullptr), "RK4 workspace upload failed");
  glasshelix::dynamics::rk4_combine_stage combine(4, 2, width, .1f);
  program::launch_binding_v2 combine_binding{workspace.value.data, combined.value.data, nullptr,
                                             workspace.value.data, 5 * width * sizeof(float)};
  program::prepared_program_v2 combine_program{2, 0, &combine.stage, 1, nullptr, 0};
  require(program::execute_prepared_program_v2(combine_program, &combine_binding, 1, nullptr)
              == program::program_status::success,
          "Cellerator RK4 combination launch failed");
  for (std::size_t i = 0; i < width; ++i)
    require(std::abs(download(combined)[i] - (base[i] + .25f)) < 1e-6f, "RK4 combination was incorrect");

  upload(current, base); upload(input, forcing);
  ce::linear_stage map_descriptor{ce::linear_kind::axpby, width, ce::device_representation::f32, .5f, .5f};
  const auto map_stage = ce::make_linear_stage(8, 3, &map_descriptor);
  program::prepared_program_v2 map_program{2, 0, &map_stage, 1, nullptr, 0};
  glasshelix::dynamics::direct_map_plan map{{map_program, {}}};
  program::launch_binding_v2 first{current.value.data, next.value.data, input.value.data};
  program::launch_binding_v2 second{next.value.data, current.value.data, input.value.data};
  require(glasshelix::dynamics::execute_resident_direct_rollout(map, first, second, 4, nullptr)
              == program::program_status::success,
          "resident repeated direct map failed");
  const auto result = download(current);
  for (std::size_t i = 0; i < width; ++i) {
    float expected = base[i];
    for (int step = 0; step < 4; ++step) expected = .5f * expected + .5f * forcing[i];
    require(std::abs(result[i] - expected) < 1e-6f, "resident map transferred state through host fallback");
  }
  return 0;
} catch (const std::exception&) {
  return 1;
}
