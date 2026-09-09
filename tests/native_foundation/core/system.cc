#include <GlassHelix/core/system.hh>
int main(){using namespace glasshelix::core;
 auto d=std::make_shared<const system_definition>("supplied",std::vector<quantity>{},std::vector<mechanism>{});
 system_instance a{d,{1},{2},{0},{0}},b{d,{3},{4},{0},{0}};
 a.state[0]=7;if(b.state[0]!=3||a.definition!=b.definition) return 1;
 bool rejected=false;try{system_definition bad("",{},{});}catch(const std::invalid_argument&){rejected=true;}
 return rejected?0:2;
}
