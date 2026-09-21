// NF1 ACCEPT complete-program measurement of the supplied GH direct-map and CE response paths.
#include <GlassHelix/dynamics/response_plan.hh>
#include <GlassHelix/inference/inference.hh>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gh = glasshelix;
namespace df = cellerator::compute::differential;
namespace ex = cellerator::execution;
namespace nf = df::nf1;
namespace nn = cellerator::compute::native_numeric;
namespace pg = ex::program;
namespace rsp = gh::dynamics::response;

constexpr unsigned raw_samples = 5;

void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
void cuda_require(cudaError_t value, const char *message) {
  if (value != cudaSuccess) throw std::runtime_error(message);
}

double host_us(const auto &call) {
  const auto begin = std::chrono::steady_clock::now();
  call();
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count();
}

struct launch_cost { double host = 0, completion = 0; };
template<class Call> launch_cost submit_timed(cudaStream_t stream, Call &&call) {
  cudaEvent_t begin{}, end{};
  cuda_require(cudaEventCreate(&begin), "create begin event");
  cuda_require(cudaEventCreate(&end), "create end event");
  cuda_require(cudaEventRecord(begin, stream), "record begin event");
  launch_cost result;
  result.host = host_us([&] { require(call() == pg::program_status::success, "prepared program launch"); });
  cuda_require(cudaEventRecord(end, stream), "record end event");
  cuda_require(cudaEventSynchronize(end), "synchronize completion event");
  float milliseconds{};
  cuda_require(cudaEventElapsedTime(&milliseconds, begin, end), "elapsed CUDA event");
  result.completion = milliseconds * 1000.0;
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return result;
}

struct device_vector {
  nn::resident_vector value{};
  void allocate(std::size_t count) {
    cuda_require(nn::allocate(&value, count, nn::device_representation::f32, 0), "allocate resident vector");
  }
  void upload(const std::vector<float> &host, cudaStream_t stream) {
    cuda_require(nn::upload(value, host.data(), host.size(), {value.generation.value + 1}, stream),
                 "upload resident vector");
  }
  void release() {
    if (value.data) cuda_require(nn::release(&value), "release resident vector");
  }
};

nf::primal_record primal(nf::identity definition, std::uint64_t state_generation,
                         std::uint64_t parameter_generation) {
  nf::primal_record result{};
  result.instance.prepared = {definition, {2, 1}, {1}, {3, 1}};
  result.instance.state = {{4, 1}, {state_generation}};
  result.instance.parameters = {{5, 1}, {parameter_generation}};
  return result;
}

struct derivative_primitive {
  std::array<nf::operand_signature, 2> inputs{};
  nf::output_signature output{};
  nf::primal_record identity{};
  df::local_block block{};
  bool unary = false;

  derivative_primitive(std::uint64_t definition, std::size_t count, cudaStream_t stream,
                       const nn::resident_vector &state, const nn::resident_vector &forcing,
                       const nn::resident_vector &parameters, bool is_unary)
      : unary(is_unary) {
    inputs = {{{{10, definition}, {}, count, ex::numeric_type::f32},
               {{11, definition}, {}, count, ex::numeric_type::f32}}};
    output = {{{12, definition}, {}, count, ex::numeric_type::f32}, {13, definition}};
    nf::operation_contract contract{};
    contract.definition = {definition, 1};
    contract.arguments = {inputs.data(), unary ? 1u : 2u};
    contract.outputs = {&output, 1};
    contract.numeric = {ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32,
                        ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32};
    contract.capabilities = nf::forward | nf::jvp | nf::vjp | nf::second_direction;
    identity = primal(contract.definition, state.generation.value, parameters.generation.value);
    const df::local_primal_owners owners{&state, unary ? nullptr : &forcing, &state, &parameters,
                                         stream, identity};
    require(df::make_local_device_block(unary ? nn::local_operation::tanh : nn::local_operation::multiply,
                                        contract, owners, block) == nf::status::success,
            "prepare local response primitive");
  }

  df::response_binding<float> jvp(const nn::resident_vector &state,
                                  const nn::resident_vector &parameters,
                                  const nn::resident_vector &left_direction,
                                  const nn::resident_vector &right_direction,
                                  nn::resident_vector &response) const {
    df::response_binding<float> binding{};
    binding.left_direction = &left_direction;
    binding.right_direction = unary ? nullptr : &right_direction;
    binding.output = &response;
    binding.count = inputs[0].element_count;
    binding.request.action = nf::jvp;
    binding.request.primal = identity;
    binding.request.primal.instance.state.generation = state.generation;
    binding.request.primal.instance.parameters.generation = parameters.generation;
    binding.request.direction_domain = inputs[0];
    binding.request.response_domain = output.operand;
    binding.direction = inputs[0];
    binding.response = output.operand;
    return binding;
  }
};

// The final GH differentiation fixture uses these prepared forward callbacks.
// They are kept here because the direct-map plan borrows its CE stage; no second engine is introduced.
struct forward_state { std::uint64_t count = 0; };
__global__ void multiply_forward_kernel(const float *state, const float *forcing,
                                        float *output, std::uint64_t count) {
  const auto index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = state[index] * forcing[index];
}
__global__ void tanh_forward_kernel(const float *state, float *output, std::uint64_t count) {
  const auto index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = tanhf(state[index]);
}
pg::program_status admit_multiply_forward(const void *prepared, const pg::launch_binding_v2 &binding,
                                          void *stream) noexcept {
  const auto *state = static_cast<const forward_state *>(prepared);
  return state && state->count && binding.input && binding.values && binding.output && stream
      ? pg::program_status::success : pg::program_status::invalid_argument;
}
pg::program_status launch_multiply_forward(const void *prepared, const pg::launch_binding_v2 &binding,
                                           void *stream) noexcept {
  if (admit_multiply_forward(prepared, binding, stream) != pg::program_status::success)
    return pg::program_status::invalid_argument;
  const auto count = static_cast<const forward_state *>(prepared)->count;
  multiply_forward_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0,
                            static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float *>(binding.input), static_cast<const float *>(binding.values),
      static_cast<float *>(binding.output), count);
  return cudaGetLastError() == cudaSuccess ? pg::program_status::success : pg::program_status::launch_failed;
}
pg::program_status admit_tanh_forward(const void *prepared, const pg::launch_binding_v2 &binding,
                                      void *stream) noexcept {
  const auto *state = static_cast<const forward_state *>(prepared);
  return state && state->count && binding.input && binding.output && stream
      ? pg::program_status::success : pg::program_status::invalid_argument;
}
pg::program_status launch_tanh_forward(const void *prepared, const pg::launch_binding_v2 &binding,
                                       void *stream) noexcept {
  if (admit_tanh_forward(prepared, binding, stream) != pg::program_status::success)
    return pg::program_status::invalid_argument;
  const auto count = static_cast<const forward_state *>(prepared)->count;
  tanh_forward_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0,
                        static_cast<cudaStream_t>(stream)>>>(
      static_cast<const float *>(binding.input), static_cast<float *>(binding.output), count);
  return cudaGetLastError() == cudaSuccess ? pg::program_status::success : pg::program_status::launch_failed;
}

gh::dynamics::prepared_stage_bundle make_forward_bundle(bool unary, forward_state &descriptor,
                                                         pg::prepared_stage_v2 &stage,
                                                         std::uint64_t stage_id,
                                                         std::uint64_t candidate_id,
                                                         const nn::resident_vector &state,
                                                         const nn::resident_vector &forcing,
                                                         nn::resident_vector &output) {
  descriptor.count = state.elements;
  stage = {stage_id, candidate_id, &descriptor, unary ? launch_tanh_forward : launch_multiply_forward,
           0, 0, 0, 0, unary ? admit_tanh_forward : admit_multiply_forward};
  pg::launch_binding_v2 binding{};
  binding.input = state.data;
  binding.output = output.data;
  binding.values = unary ? nullptr : forcing.data;
  return {{2, 0, &stage, 1, nullptr, 0}, {binding}};
}

enum class activity_kind { all, half, churn };
const char *activity_name(activity_kind value) {
  switch (value) { case activity_kind::all: return "all"; case activity_kind::half: return "half"; case activity_kind::churn: return "churn"; }
  return "invalid";
}

struct phase_cost {
  double allocation = 0, initial_upload = 0, prepare = 0;
  double value_refresh = 0, activity_refresh = 0;
  double forward_launch = 0, forward_completion = 0;
  double response_launch = 0, response_completion = 0;
  double observation = 0, teardown = 0, total = 0;
};
void add(phase_cost &to, const phase_cost &from) {
  to.value_refresh += from.value_refresh; to.activity_refresh += from.activity_refresh;
  to.forward_launch += from.forward_launch; to.forward_completion += from.forward_completion;
  to.response_launch += from.response_launch; to.response_completion += from.response_completion;
  to.observation += from.observation;
}

struct candidate_use {
  phase_cost cost{};
  gh::observation::engine_output output{};
};

struct candidate_context {
  std::size_t count{};
  bool unary{};
  std::uint64_t candidate_id{};
  device_vector state, forcing, parameters, left_direction, right_direction, output, response;
  forward_state forward_descriptor{};
  pg::prepared_stage_v2 forward_stage{};
  gh::dynamics::prepared_stage_bundle forward{};
  std::unique_ptr<derivative_primitive> derivative;
  ex::value_generation activity_generation{1};
  gh::dynamics::response_attachment attachment{};
  std::unique_ptr<gh::dynamics::direct_map_plan> plan;
  rsp::saved_primal saved{};
  std::vector<float> state_host, forcing_host, left_direction_host, right_direction_host, activity_host;

  candidate_context(std::size_t width, bool is_unary, std::uint64_t id)
      : count(width), unary(is_unary), candidate_id(id) {}

  void allocate() {
    for (auto *vector : {&state, &forcing, &parameters, &left_direction, &right_direction, &output, &response})
      vector->allocate(count);
  }

  void make_values(activity_kind activity, unsigned sample, unsigned step, unsigned direction) {
    activity_host.assign(count, 1.f);
    for (std::size_t index = 0; index < count; ++index) {
      const bool active = activity == activity_kind::all ||
          (activity == activity_kind::half ? index % 2 == 0 : (index + sample + step) % 2 == 0);
      activity_host[index] = active ? 1.f : 0.f;
    }
    state_host.resize(count); forcing_host.resize(count);
    left_direction_host.resize(count); right_direction_host.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
      const float coefficient = activity_host[index];
      state_host[index] = coefficient * (.20f + .01f * float(index) + .005f * float(candidate_id) + .0005f * float(sample + step));
      forcing_host[index] = coefficient * (.35f + .007f * float(index) + .003f * float(candidate_id));
      left_direction_host[index] = coefficient * (direction == 0 ? .05f : 0.f);
      right_direction_host[index] = coefficient * (unary ? 0.f : (direction == 1 ? .07f : 0.f));
    }
  }

  void initial_upload(activity_kind activity, cudaStream_t stream_value) {
    make_values(activity, 0, 0, 0);
    state.upload(state_host, stream_value); forcing.upload(forcing_host, stream_value);
    parameters.upload(std::vector<float>(count, 1.f), stream_value);
    left_direction.upload(left_direction_host, stream_value); right_direction.upload(right_direction_host, stream_value);
    cuda_require(cudaStreamSynchronize(stream_value), "initial upload completion");
  }

  void prepare(cudaStream_t stream_value) {
    derivative = std::make_unique<derivative_primitive>(1000 + candidate_id, count, stream_value,
                                                         state.value, forcing.value, parameters.value, unary);
    forward = make_forward_bundle(unary, forward_descriptor, forward_stage, 4000 + candidate_id,
                                  candidate_id, state.value, forcing.value, output.value);
    attachment = {{4000 + candidate_id, candidate_id}, &derivative->block, {},
                  {unary ? nullptr : &forcing.value}, &activity_generation};
    plan = std::make_unique<gh::dynamics::direct_map_plan>(forward, gh::core::differentiated_object::vector_field, attachment);
    saved = rsp::retain_primal(attachment);
    require(saved.state_owner != nullptr, "retain initial GH direct-map primal");
  }

  candidate_use use(activity_kind activity, unsigned sample, unsigned step, unsigned direction,
                    cudaStream_t stream_value) {
    candidate_use result{};
    const auto before_map = activity_host;
    result.cost.value_refresh = host_us([&] {
      make_values(activity, sample, step, direction);
      state.upload(state_host, stream_value); forcing.upload(forcing_host, stream_value);
      left_direction.upload(left_direction_host, stream_value); right_direction.upload(right_direction_host, stream_value);
      cuda_require(cudaStreamSynchronize(stream_value), "value refresh completion");
    });
    result.cost.activity_refresh = host_us([&] {
      if (activity_host != before_map) ++activity_generation.value;
      saved = rsp::retain_primal(attachment);
      require(saved.state_owner != nullptr, "retain refreshed GH direct-map primal");
    });
    const auto forward_cost = submit_timed(stream_value, [&] { return gh::dynamics::execute_direct_map(*plan, stream_value); });
    result.cost.forward_launch = forward_cost.host;
    result.cost.forward_completion = forward_cost.completion;
    auto binding = derivative->jvp(state.value, parameters.value, left_direction.value,
                                   right_direction.value, response.value);
    const auto response_cost = submit_timed(stream_value, [&] {
      return rsp::execute_direct_map_jvp(gh::core::differentiated_object::vector_field,
                                         *plan, saved, binding, stream_value);
    });
    result.cost.response_launch = response_cost.host;
    result.cost.response_completion = response_cost.completion;
    result.cost.observation = host_us([&] { result.output = observe_and_verify(stream_value, direction); });
    return result;
  }

  gh::observation::engine_output observe_and_verify(cudaStream_t stream_value, unsigned direction) const {
    std::vector<float> forward_host(count), response_host(count);
    cuda_require(nn::download(output.value, forward_host.data(), count, stream_value), "download GH forward output");
    cuda_require(nn::download(response.value, response_host.data(), count, stream_value), "download GH response output");
    cuda_require(cudaStreamSynchronize(stream_value), "observation download completion");
    for (std::size_t index = 0; index < count; ++index) {
      const float expected_forward = unary ? std::tanh(state_host[index]) : state_host[index] * forcing_host[index];
      const float expected_response = unary
          ? (1.f - expected_forward * expected_forward) * left_direction_host[index]
          : forcing_host[index] * left_direction_host[index] + state_host[index] * right_direction_host[index];
      require(std::abs(forward_host[index] - expected_forward) < 2e-6f, "full forward output mismatch");
      require(std::abs(response_host[index] - expected_response) < 3e-6f, "full analytic JVP mismatch");
    }
    std::vector<double> values(forward_host.begin(), forward_host.end());
    gh::core::result_provenance provenance{};
    provenance.scientific_model_id = "gh-nf1a-direct-map";
    provenance.model_revision = unary ? "tanh" : "multiply";
    provenance.gh_source_commit = "4df14422";
    provenance.ce_source_commit = "2c3b094";
    provenance.numerical_policy = "fp32";
    require(provenance.valid(), "direct-map provenance");
    return {std::move(values), std::move(provenance)};
  }

  void release() {
    plan.reset();
    derivative.reset();
    for (auto *vector : {&response, &output, &right_direction, &left_direction, &parameters, &forcing, &state}) vector->release();
  }
};

struct fixture {
  std::size_t width{};
  bool unary{};
  int candidates{};
  cudaStream_t stream{};
  std::vector<std::unique_ptr<candidate_context>> contexts;

  fixture(std::size_t count, bool is_unary, int supplied) : width(count), unary(is_unary), candidates(supplied) {}
  void allocate() {
    cuda_require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create benchmark stream");
    for (int index = 0; index < candidates; ++index) {
      auto context = std::make_unique<candidate_context>(width, unary, index + 1);
      context->allocate(); contexts.push_back(std::move(context));
    }
  }
  void initial_upload(activity_kind activity) {
    for (auto &context : contexts) {
      context->initial_upload(activity, stream);
    }
  }
  void prepare() {
    for (auto &context : contexts) context->prepare(stream);
  }
  phase_cost use(activity_kind activity, unsigned sample, unsigned steps, unsigned directions) {
    phase_cost result{};
    std::vector<gh::observation::engine_output> outputs;
    for (unsigned step = 0; step < steps; ++step) {
      for (unsigned direction = 0; direction < directions; ++direction) {
        outputs.clear();
        for (auto &context : contexts) {
          auto executed = context->use(activity, sample, step, direction, stream);
          add(result, executed.cost);
          outputs.push_back(std::move(executed.output));
        }
        result.observation += host_us([&] { observe_candidates(outputs, sample, step, direction); });
      }
    }
    return result;
  }
  void observe_candidates(const std::vector<gh::observation::engine_output> &outputs,
                          unsigned sample, unsigned step, unsigned direction) const {
    gh::core::quantity observed{};
    observed.axis = {{gh::core::identity::biological_abi_version,
                      gh::core::identity::serialized_record_kind::persistent_axis_identity,
                      sizeof(gh::core::identity::persistent_axis_identity)},
                     {1, 0}, {2, 0}, {3, 0}, {4, 0}};
    observed.structure = {5, 0}; observed.units = "direct-map-output";
    observed.role = gh::core::quantity_role::observation; observed.extent = 2;
    const auto map = gh::observation::observation_map::partial(observed, width, {0, width - 1});
    std::vector<gh::inference::candidate> supplied;
    supplied.reserve(outputs.size());
    for (std::size_t index = 0; index < outputs.size(); ++index)
      supplied.push_back({"candidate-" + std::to_string(index + 1), unary ? "tanh" : "multiply",
                          {outputs[index].values.front(), outputs[index].values.back()}, 1.0});
    gh::inference::finite_candidates belief{std::move(supplied)};
    belief.propagate([&](const gh::inference::candidate &candidate) {
      const auto index = static_cast<std::size_t>(std::stoul(candidate.hypothesis_id.substr(10)) - 1);
      return outputs.at(index);
    });
    const auto predicted = map.evaluate(outputs.front());
    const gh::core::observation_record evidence{observed, predicted, {true, true}, "direct-map",
                                                  "steps", "accept", "declared", double(step),
                                                  gh::core::sampling_unit::population_snapshot, true};
    belief.assimilate(evidence, gh::observation::gaussian_noise{.1, sample + direction + 1}, map,
                       "accept-" + std::to_string(sample) + "-" + std::to_string(step) + "-" + std::to_string(direction));
    require(belief.values().size() == outputs.size(), "observe supplied candidates");
  }
  double teardown() {
    return host_us([&] {
      if (stream) cuda_require(cudaStreamSynchronize(stream), "teardown synchronization");
      for (auto &context : contexts) context->release();
      contexts.clear();
      if (stream) cuda_require(cudaStreamDestroy(stream), "destroy benchmark stream");
      stream = nullptr;
    });
  }
  ~fixture() { if (stream) { for (auto &context : contexts) context->release(); cudaStreamDestroy(stream); } }
};

struct scenario {
  std::size_t width; bool unary; int candidates; activity_kind activity; unsigned steps; unsigned directions;
};

struct row { std::string mode; scenario scenario_value{}; unsigned sample{}; phase_cost cost{}; };
double median(std::vector<double> values) { std::sort(values.begin(), values.end()); return values[values.size() / 2]; }
void print_row(const row &value) {
  const auto &s = value.scenario_value; const auto &p = value.cost;
  std::printf("%s,%zu,%s,%d,%s,%u,%u,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
              value.mode.c_str(), s.width, s.unary ? "unary_tanh" : "binary_multiply", s.candidates,
              activity_name(s.activity), s.steps, s.directions, value.sample, p.allocation, p.initial_upload,
              p.prepare, p.value_refresh, p.activity_refresh, p.forward_launch, p.forward_completion,
              p.response_launch, p.response_completion, p.observation, p.teardown, p.total);
}

int main() {
  try {
    cuda_require(cudaSetDevice(0), "select CUDA device");
    std::puts("mode,width,arity,candidates,activity_realization,steps,directions,sample,allocation_us,initial_upload_us,prepare_us,value_refresh_us,activity_refresh_us,forward_launch_us,forward_completion_us,response_launch_us,response_completion_us,observation_us,teardown_us,total_us");
    const std::array<scenario, 5> scenarios{{
        {16, false, 1, activity_kind::all, 1, 2},
        {33, true, 1, activity_kind::half, 1, 1},
        {33, false, 4, activity_kind::half, 4, 2},
        {65, true, 4, activity_kind::churn, 4, 1},
        {65, false, 4, activity_kind::churn, 4, 2},
    }};
    for (const auto &scenario_value : scenarios) {
      std::vector<row> cold, resident;
      for (unsigned sample = 0; sample < raw_samples; ++sample) {
        fixture item{scenario_value.width, scenario_value.unary, scenario_value.candidates};
        row value{"cold", scenario_value, sample};
        const auto all = std::chrono::steady_clock::now();
        value.cost.allocation = host_us([&] { item.allocate(); });
        value.cost.initial_upload = host_us([&] { item.initial_upload(scenario_value.activity); });
        value.cost.prepare = host_us([&] { item.prepare(); });
        const auto use = item.use(scenario_value.activity, sample, scenario_value.steps, scenario_value.directions);
        add(value.cost, use);
        value.cost.teardown = item.teardown();
        value.cost.total = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - all).count();
        cold.push_back(value); print_row(value);
      }
      fixture item{scenario_value.width, scenario_value.unary, scenario_value.candidates};
      const auto setup_all = std::chrono::steady_clock::now();
      const double allocation = host_us([&] { item.allocate(); });
      const double initial_upload = host_us([&] { item.initial_upload(scenario_value.activity); });
      const double preparation = host_us([&] { item.prepare(); });
      const double setup = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - setup_all).count();
      for (unsigned sample = 0; sample < raw_samples; ++sample) {
        row value{"resident", scenario_value, sample};
        const auto use_all = std::chrono::steady_clock::now();
        const auto use = item.use(scenario_value.activity, sample, scenario_value.steps, scenario_value.directions);
        add(value.cost, use);
        value.cost.allocation = allocation / raw_samples;
        value.cost.initial_upload = initial_upload / raw_samples;
        value.cost.prepare = preparation / raw_samples;
        value.cost.total = setup / raw_samples + std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - use_all).count();
        resident.push_back(value);
      }
      const double teardown = item.teardown();
      for (auto &value : resident) { value.cost.teardown = teardown / raw_samples; value.cost.total += value.cost.teardown; print_row(value); }
      for (const auto &[name, values] : std::array<std::pair<const char *, const std::vector<row> *>, 2>{{{"cold_median", &cold}, {"resident_median", &resident}}}) {
        auto collect = [&](auto member) { std::vector<double> result; for (const auto &entry : *values) result.push_back(member(entry.cost)); return median(std::move(result)); };
        row value{name, scenario_value, raw_samples};
        value.cost = {collect([](const phase_cost &p) { return p.allocation; }), collect([](const phase_cost &p) { return p.initial_upload; }), collect([](const phase_cost &p) { return p.prepare; }), collect([](const phase_cost &p) { return p.value_refresh; }), collect([](const phase_cost &p) { return p.activity_refresh; }), collect([](const phase_cost &p) { return p.forward_launch; }), collect([](const phase_cost &p) { return p.forward_completion; }), collect([](const phase_cost &p) { return p.response_launch; }), collect([](const phase_cost &p) { return p.response_completion; }), collect([](const phase_cost &p) { return p.observation; }), collect([](const phase_cost &p) { return p.teardown; }), collect([](const phase_cost &p) { return p.total; })};
        print_row(value);
      }
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
