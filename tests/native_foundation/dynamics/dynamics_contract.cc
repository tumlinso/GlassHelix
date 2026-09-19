#include <GlassHelix/dynamics/evolution.hh>
#include <stdexcept>
using namespace glasshelix;
int main() {
  core::rk4_request request; request.begin=0; request.end=2; request.dt=.01;
  request.forcing.knots={{0,{1}},{.75,{0}}};
  const auto intervals=dynamics::split_forcing_intervals(request);
  if(intervals.size()!=2 || intervals[0].begin!=0 || intervals[0].end!=.75 || intervals[1].begin!=.75 || intervals[1].knot_index!=1) return 1;
  const auto schedule=dynamics::make_rk4_schedule(request);
  if(schedule.size()!=2 || schedule[0].steps!=75 || schedule[1].steps!=125
     || schedule[0].forcing.knot_index!=0 || schedule[1].forcing.knot_index!=1) return 5;
  auto off_grid=request; off_grid.forcing.knots[1].time=.751;
  bool rejected_discontinuity=false;
  try { (void)dynamics::make_rk4_schedule(off_grid); } catch(const std::invalid_argument&) { rejected_discontinuity=true; }
  if(!rejected_discontinuity) return 6;
  dynamics::resident_instance instance{{1,0},{4},{7},"cellerator-policy"};
  dynamics::bounded_primal_history history(1);
  history.record({instance.identity,instance.state_generation,instance.parameter_generation,.75,75,reinterpret_cast<void*>(1)});
  if(!dynamics::valid_checkpoint(history.require_current(instance,75),instance)) return 2;
  instance.parameter_generation.value++;
  bool rejected=false;
  try { history.require_current(instance,75); } catch(const std::invalid_argument&) { rejected=true; }
  if(!rejected) return 3;
  instance.parameter_generation.value--;
  instance.state_generation.value++;
  rejected=false;
  try { history.require_current(instance,75); } catch(const std::invalid_argument&) { rejected=true; }
  if(!rejected) return 7;
  dynamics::rk4_combine_stage combine(4,2,33,.01f);
  return combine.stage.launch && combine.stage.prepared_state==&combine.descriptor ? 0 : 4;
}
