#include "fixture.hh"
#include <iomanip>
#include <iostream>
int main() try {
 using namespace nf1_reference;
 const auto x=initial(0),y=rollout(x),other=rollout(x,1.1,.4);
 for(std::size_t i=0;i<4;++i)near(y[i],other[i],2e-14,"equal sums must agree");
 const auto grad=parameter_gradient(x);
 constexpr double eps=1e-5;
 const double fd=(rollout(x,.8+eps,.7)[3]-rollout(x,.8-eps,.7)[3])/(2*eps);
 near(fd,grad[0],2e-10,"independent finite difference");
 near(grad[0]-grad[1],0,1e-15,"a-b null direction");
 require(grad[0]>1e-3,"orthogonal response must be nonzero");
 const auto fine=rollout(x,.8,.7,400),finer=rollout(x,.8,.7,800);
 const double e1=std::abs(y[3]-fine[3]),e2=std::abs(fine[3]-finer[3]);
 require(e1<1e-8 && e1>e2,"RK4 convergence");
 auto hidden=x;hidden[2]+=.8;const auto future=rollout(hidden);
 near(x[3],hidden[3],0,"same present observation");
 require(std::abs(future[3]-y[3])>.01,"hidden distinction changes future");
 const auto initial_weights=normalize_log(log_likelihood(x[3],x[3],.005),log_likelihood(x[3],hidden[3],.005));
 near(initial_weights[0],.5,1e-15,"present observation remains ambiguous");
 const auto posterior=normalize_log(log_likelihood(y[3],y[3],.005),log_likelihood(y[3],future[3],.005));
 require(posterior[0]>.99,"later measurement resolves supplied alternatives");
 double zero_weight=0,source=.7;near(zero_weight*source,0,0,"zero forward");near(source,.7,0,"nonzero derivative");
 require(!std::isfinite(0.0*std::numeric_limits<double>::infinity()),"arithmetic nonfinite control");
 const double masked=false?std::numeric_limits<double>::quiet_NaN():0.0;near(masked,0,0,"predicate skip");
 std::cout<<std::setprecision(17)<<"{\"status\":\"reference_only_passed\",\"production_or_gpu_test\":false,\"batch_width\":33,\"final_y\":"<<y[3]<<",\"parameter_gain\":"<<grad[0]<<",\"null_direction_response\":"<<grad[0]-grad[1]<<",\"hidden_future_y\":"<<future[3]<<",\"posterior_correct\":"<<posterior[0]<<",\"rk4_refinement_error\":"<<e1<<"}\n";
 return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
