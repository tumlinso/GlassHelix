#pragma once
#include <cmath>
#include <string>
#include <variant>
#include <vector>
namespace glasshelix::core {
struct forcing_knot { double time; std::vector<double> values; };
struct forcing_schedule {
 std::string time_units;
 std::vector<forcing_knot> knots;
 bool valid()const {
  if(time_units.empty())return false;
  double previous=-INFINITY;
  for(const auto& k:knots){if(!std::isfinite(k.time)||k.time<=previous)return false;previous=k.time;}
  return true;
 }
};
struct forward_request {};
struct discrete_request { std::size_t steps=1; }; // No dt multiplier exists.
struct rk4_request { double begin=0,end=1,dt=0.1; std::string time_units="dimensionless"; forcing_schedule forcing{"dimensionless",{}}; };
using evolution_request=std::variant<forward_request,discrete_request,rk4_request>;
enum class differentiated_object { vector_field, discrete_step, observation, implemented_rollout };
struct response_request { differentiated_object object; evolution_request evolution; std::vector<std::size_t> selected_outputs; };
struct observation_request { evolution_request evolution; std::vector<std::size_t> selected_outputs; };
struct comparison_request { evolution_request evolution; std::vector<std::string> supplied_system_ids; };
inline bool valid(const rk4_request& r) {
 return std::isfinite(r.begin)&&std::isfinite(r.end)&&std::isfinite(r.dt)&&r.end>=r.begin&&r.dt>0
  &&!r.time_units.empty()&&r.forcing.time_units==r.time_units&&r.forcing.valid();
}
// RK4 runtime must split at forcing discontinuities and finish each interval
// using its preceding forcing value, including the endpoint stage.
}
