#include <GlassHelix/core/host_bindings.hh>
#include <limits>
#include <stdexcept>
namespace glasshelix::core {
host_binding_dimensions required_host_dimensions(const system_definition& definition) {
  host_binding_dimensions dimensions{0, 0};
  for (const auto& q : definition.quantities()) {
    std::size_t* count = nullptr;
    if (q.role == quantity_role::dynamical_state) count = &dimensions.state;
    if (q.role == quantity_role::parameter) count = &dimensions.parameters;
    if (!count) continue;
    if (q.extent > std::numeric_limits<std::size_t>::max() - *count)
      throw std::overflow_error("scientific binding extent overflow");
    *count += q.extent;
  }
  return dimensions;
}
system_instance bind_host_instance(std::shared_ptr<const system_definition> definition,
                                  std::span<const double> state,
                                  std::span<const double> parameters) {
  if (!definition) throw std::invalid_argument("scientific definition required");
  const auto expected = required_host_dimensions(*definition);
  if (state.size() != expected.state || parameters.size() != expected.parameters)
    throw std::invalid_argument("state/parameter bindings do not match declared quantity roles");
  return {std::move(definition), {state.begin(), state.end()},
          {parameters.begin(), parameters.end()}, {0}, {0}};
}
}
