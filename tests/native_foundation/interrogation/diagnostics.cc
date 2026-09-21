#include <GlassHelix/interrogation/diagnostics.hh>
#include <cmath>
int main(){auto r=glasshelix::interrogation::rank_revealing_svd({1.,1.,2.,2.},2,2,{1.,1.},1e-10,"local linearization");return r.rank==1&&r.singular_values[0]>0&&r.singular_values[1]<1e-10?0:1;}
