#include <GlassHelix/core/capability.hh>
int main(){using namespace glasshelix::core;extension_dispatch d;
 for(auto k:{extension_kind::history,extension_kind::stochastic_transition,extension_kind::jvp,extension_kind::vjp,extension_kind::second_direction})if(d.invoke(k)!=execution_status::unsupported)return 1;
 bool rejected=false;try{d.register_implementation(extension_kind::deterministic_forward,{});}catch(const std::invalid_argument&){rejected=true;}
 return rejected?0:2;
}
