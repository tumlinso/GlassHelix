#include <GlassHelix/interrogation/diagnostics.hh>
#include <array>
int main(){const std::array<double,4>a{1.,1.,2.,2.};const std::array<double,2>s{1.,1.};auto r=glasshelix::interrogation::rank_revealing_svd(a,2,2,s,1e-10,"local linearization");return r.rank==1&&r.singular_values[0]>0&&r.singular_values[1]<1e-10?0:1;}
