#include <GlassHelix/core/request.hh>
int main() {
  using namespace glasshelix::core;
  rk4_request request; if(!valid(request)) return 1;
  request.dt=0; if(valid(request)) return 2; request.dt=.1;
  request.forcing.knots={{.75,{1}},{.75,{2}}}; if(valid(request)) return 3;
  request.forcing.knots={{0,{1}},{.75,{2}}}; return valid(request) ? 0 : 4;
}
