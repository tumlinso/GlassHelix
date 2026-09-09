#pragma once
#include <GlassHelix/core/system.hh>
#include <span>
namespace glasshelix::core {
struct host_binding_dimensions { std::size_t state; std::size_t parameters; };
// Role-aware host scientific binding. This does not allocate device state or
// replace Cellerator instance bindings. Other roles require explicit owners.
host_binding_dimensions required_host_dimensions(const system_definition&);
system_instance bind_host_instance(std::shared_ptr<const system_definition>,
                                  std::span<const double> state,
                                  std::span<const double> parameters);
}
