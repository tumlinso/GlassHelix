#pragma once
#include <Cellerator/execution/identity.hh>
#include <cmath>
#include <optional>
#include <string>
#include <vector>
namespace glasshelix::core {
namespace identity = cellerator::execution;
enum class quantity_role { dynamical_state, observation, inferential_information, parameter, forcing, context };
struct quantity {
  identity::persistent_axis_identity axis{};
  identity::structure_id structure{};
  identity::structure_epoch epoch{};
  quantity_role role = quantity_role::dynamical_state;
  std::size_t extent = 0;
  std::string units;
  double scale = 1;
  std::optional<std::string> biological_label;
};
inline bool valid(const quantity& q) {
  return identity::validate_persistent_axis_identity(q.axis) == identity::biological_validation_code::ok
    && identity::valid_identity(q.structure) && !q.units.empty() && std::isfinite(q.scale) && q.scale > 0;
}
inline bool interchangeable(const quantity& a, const quantity& b) {
  return valid(a) && valid(b) && a.role == b.role && a.extent == b.extent
    && a.units == b.units && a.scale == b.scale && a.epoch.value == b.epoch.value
    && identity::same_identity(a.structure,b.structure)
    && identity::same_identity(a.axis.domain,b.axis.domain)
    && identity::same_identity(a.axis.order,b.axis.order)
    && identity::same_identity(a.axis.geometry,b.axis.geometry)
    && identity::same_identity(a.axis.partition,b.axis.partition);
}
}
