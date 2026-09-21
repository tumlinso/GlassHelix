#include <GlassHelix/interrogation/alternatives.hh>
int main(){using namespace glasshelix::interrogation;alternative a{"a","reporter",{1,2},{3}},b{"b","reporter",{1,2},{4}};return aligned(a,b)&&exclusive_branch(a,b)?0:1;}
