#include <GlassHelix/core/request.hh>
int main(){using namespace glasshelix::core;rk4_request r;if(!valid(r))return 1;
 r.dt=0;if(valid(r))return 2;r.dt=.1;r.forcing.knots={{.75,{1}},{.75,{2}}};if(valid(r))return 3;
 evolution_request d=discrete_request{2};if(!std::holds_alternative<discrete_request>(d))return 4;
 r.forcing.knots={{0,{1}},{.75,{2}}};return valid(r)?0:5;
}
