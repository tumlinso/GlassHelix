#include <GlassHelix/reduction/contracts.hh>
int main(){using namespace glasshelix::reduction;validate({.1,.2,.3,true},.3);return equal_decay_closed(1,2,.5)&&preserves_reporter_correlation(1,1)&&!preserves_reporter_correlation(1,2)?0:1;}
