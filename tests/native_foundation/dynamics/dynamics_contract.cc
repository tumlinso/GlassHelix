#include <GlassHelix/dynamics/evolution.hh>
#include <stdexcept>
using namespace glasshelix;
int main() {
  core::rk4_request request; request.begin=0; request.end=2; request.dt=.01;
  request.forcing.knots={{0,{1}},{.75,{0}}};
  const auto intervals=dynamics::split_forcing_intervals(request);
  if(intervals.size()!=2 || intervals[0].begin!=0 || intervals[0].end!=.75 || intervals[1].begin!=.75 || intervals[1].knot_index!=1) return 1;
  dynamics::resident_instance instance{{1,0},{4},{7},"cellerator-policy"};
  dynamics::bounded_primal_history history(1);
  history.record({instance.identity,instance.state_generation,instance.parameter_generation,.75,75,reinterpret_cast<void*>(1)});
  if(!dynamics::valid_checkpoint(history.require_current(instance,75),instance)) return 2;
  instance.parameter_generation.value++;
  try { history.require_current(instance,75); } catch(const std::invalid_argument&) { return 0; }
  return 3;
}
