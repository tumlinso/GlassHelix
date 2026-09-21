#include <GlassHelix/interrogation/diagnostics.hh>
#include <array>
#include <stdexcept>
int main(){using glasshelix::interrogation::rank_revealing_svd;const std::array<double,4>a{1.,1.,2.,2.}, full{1.,0.,0.,1.};const std::array<double,2>s{1.,1.};auto u=rank_revealing_svd(a,2,2,s,1e-10,"local");auto r=rank_revealing_svd(full,2,2,s,1e-10,"control");if(u.rank!=1||u.right_nullspace.size()!=2||u.local_domain!="local"||r.rank!=2||!r.right_nullspace.empty())return 1;bool bad=false;try{rank_revealing_svd(a,2,2,{1.},1e-10,"x");}catch(...){bad=true;}return bad?0:2;}
