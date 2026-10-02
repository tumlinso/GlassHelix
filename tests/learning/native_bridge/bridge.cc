#include <GlassHelix/core/system.hh>
#include <GlassHelix/observation/observation.hh>
#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace gh = glasshelix;
namespace nn = cellerator::compute::native_numeric;
namespace df = cellerator::compute::differential;

static void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main(int argc, char** argv) try {
  require(argc == 2, "usage: gh_ml2_native_bridge bridge-values.txt");
  std::ifstream input(argv[1]);
  require(input.good(), "cannot read fitted bridge values");
  std::map<std::string, std::string> fields;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    std::string key, value, extra;
    require(bool(row >> key >> value) && !(row >> extra), "expected key value per line");
    require(fields.emplace(key, value).second, "duplicate input key");
  }
  auto text = [&](const std::string& key) { return fields.at(key); };
  auto scalar = [&](const std::string& key) {
    const auto raw = text(key); std::size_t end = 0;
    const double value = std::stod(raw, &end);
    require(end == raw.size() && std::isfinite(value), "nonfinite or invalid scalar");
    return value;
  };
  auto integer = [&](const std::string& key) {
    const auto raw = text(key); std::size_t end = 0;
    require(!raw.empty() && raw[0] != '-', "negative identity");
    const auto value = std::stoull(raw, &end);
    require(end == raw.size(), "invalid identity");
    return value;
  };
  const double k = scalar("shared_coefficient"), x = scalar("input_x"),
    z = scalar("local_z"), forcing = scalar("forcing");
  gh::core::quantity quantity;
  quantity.axis = {{gh::core::identity::biological_abi_version,
      gh::core::identity::serialized_record_kind::persistent_axis_identity,
      sizeof(gh::core::identity::persistent_axis_identity)},
      {integer("domain_id"), integer("domain_id") + 1000}, {integer("order_id"), integer("order_id") + 1000},
      {integer("geometry_id"), integer("geometry_id") + 1000}, {integer("partition_id"), integer("partition_id") + 1000}};
  quantity.structure = {integer("structure_id"), 0};
  quantity.epoch = {integer("structure_epoch")};
  quantity.extent = 1; quantity.units = text("units"); quantity.scale = scalar("scale");
  auto qx = quantity, qk = quantity, qz = quantity, qf = quantity, qr = quantity;
  qx.extent = integer("source_extent"); qz.extent = qx.extent;
  qr.axis.domain = {integer("target_domain_id"), integer("target_domain_id") + 1000};
  qr.axis.order = {integer("target_order_id"), integer("target_order_id") + 1000};
  qr.axis.geometry = {integer("target_geometry_id"), integer("target_geometry_id") + 1000};
  qr.axis.partition = {integer("target_partition_id"), integer("target_partition_id") + 1000};
  qr.extent = integer("target_extent");
  qk.axis.domain = {integer("parameter_domain_id"), integer("parameter_domain_id") + 1000};
  qk.axis.order = {integer("parameter_order_id"), integer("parameter_order_id") + 1000};
  qk.axis.geometry = {integer("parameter_geometry_id"), integer("parameter_geometry_id") + 1000};
  qk.axis.partition = {integer("parameter_partition_id"), integer("parameter_partition_id") + 1000};
  require(qx.extent == 2 && qr.extent == 1, "fitted model axes changed");
  require(text("hypothesis_label") == "synthetic-product-7001-v1" &&
          text("model_id") == "7001" && text("mechanism_id") == "501",
          "fitted hypothesis mapping changed");
  qx.role = gh::core::quantity_role::dynamical_state;
  qk.role = gh::core::quantity_role::parameter;
  qz.role = gh::core::quantity_role::inferential_information;
  qf.role = gh::core::quantity_role::forcing;
  qr.role = gh::core::quantity_role::observation;
  const gh::core::system_definition model(text("model_id"), {qx, qk, qz, qf, qr},
    {{text("mechanism_id"), {1, 0, 2, 3},
      {"shared_coefficient", "state", "local_unknown", "forcing"}, {4}, false}});
  std::array<double, 1> ak{k}, ax{x}, az{z}, af{forcing}, zero{0}, one{1},
      kx{}, kxz{}, result{}, dkxz{}, derivative{};
  auto multiply = [&](auto& left, auto& right, auto& output) {
    require(nn::local_forward(nn::local_operation::multiply,
      std::span<const double>(left), std::span<const double>(right),
      std::span<double>(output)) == nn::local_status::success, "CE primal failed");
  };
  multiply(ak, ax, kx); multiply(kx, az, kxz); multiply(kxz, af, result);
  df::local_binding<double> inner{kx, az, zero, one, {}, dkxz, {}, {}};
  require(df::local_jvp(nn::local_operation::multiply, inner) == nn::local_status::success,
          "CE inner JVP failed");
  df::local_binding<double> outer{kxz, af, dkxz, zero, {}, derivative, {}, {}};
  require(df::local_jvp(nn::local_operation::multiply, outer) == nn::local_status::success,
          "CE outer JVP failed");
  gh::core::result_provenance provenance;
  provenance.scientific_model_id = model.id(); provenance.model_revision = text("model_revision");
  provenance.gh_source_commit = GH_SOURCE_COMMIT; provenance.ce_source_commit = CE_SOURCE_COMMIT;
  provenance.structure_epoch = quantity.epoch;
  provenance.state_generation = {integer("state_generation")};
  provenance.parameter_generation = {integer("parameter_generation")};
  provenance.units = {quantity.units}; provenance.scales = {quantity.scale};
  provenance.numerical_policy = "CE host FP64 multiplication; FP32 fitted checkpoint comparison";
  provenance.numerical_approximations = {{gh::core::approximation_kind::exact_declared_arithmetic,
      "CE local arithmetic", "supplied fitted scalar values", {}}};
  const auto observation = gh::observation::observation_map::partial(qr, 1, {0});
  const auto output = observation.evaluate({{result[0]}, provenance});
  const auto response = observation.evaluate({{derivative[0]}, provenance});
  const double tolerance = scalar("tolerance");
  require(tolerance > 0 && tolerance <= 1e-3, "invalid comparison tolerance");
  auto close = [&](double actual, double expected) {
    return std::abs(actual - expected) <= tolerance * std::max(1.0, std::abs(expected));
  };
  require(close(output[0], scalar("expected_output")), "fitted output differs");
  require(close(response[0], scalar("derivative_z")), "fitted selected response differs");
  require(integer("observed_mask") <= 1, "invalid observation mask");
  gh::core::observation_record record{qr, output, {integer("observed_mask") == 1},
      text("modality"), text("time_units"), text("sampling_id"),
      "synthetic engineering fixture", scalar("time"),
      gh::core::sampling_unit::population_snapshot, true};
  require(record.valid(), "invalid typed observation record");
  require(gh::core::interchangeable(qr, observation.output()), "observation identity changed");
  std::cout << std::setprecision(17) << "{\"status\":\"pass\",\"output\":" << output[0]
    << ",\"derivative_z\":" << response[0] << ",\"gh_source_commit\":\"" << GH_SOURCE_COMMIT
    << "\",\"ce_source_commit\":\"" << CE_SOURCE_COMMIT << "\"}\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n'; return 1;
}
