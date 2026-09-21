#include <GlassHelix/core/system.hh>
int main() {
  using namespace glasshelix::core;
  auto definition=std::make_shared<const system_definition>("supplied",std::vector<quantity>{},std::vector<mechanism>{});
  system_instance first{definition,{1,0},{7},{11}}, second{definition,{2,0},{8},{12}};
  if(first.definition!=second.definition || first.instance.low==second.instance.low || first.state_generation.value==second.state_generation.value) return 1;
  bool rejected=false; try { system_definition bad("",{},{}); } catch(const std::invalid_argument&) { rejected=true; }
  return rejected ? 0 : 2;
}
