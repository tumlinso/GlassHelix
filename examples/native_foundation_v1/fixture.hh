#pragma once
// Independent scalar formulas. This file must never be used as a production backend.
#include <array>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <vector>
namespace nf1_reference {
using state=std::array<double,4>;
inline double forcing(double t) { return t < 0.75 ? 1.0 : 0.0; }
inline double triad(const state& x) { return std::tanh(x[0])*(x[1]/(1.0+x[1]*x[1]))*x[2]; }
inline state rhs(const state& x,double a,double b,double u) {
 return {u-.7*x[0],.4*x[0]-.5*x[1],.2+.3*x[1]-.4*x[2],(a+b)*triad(x)-.6*x[3]};
}
inline state shifted(const state& x,const state& k,double h) {
 state out{};for(std::size_t i=0;i<4;++i)out[i]=x[i]+h*k[i];return out;
}
inline state step(const state& x,double a,double b,double u,double h) {
 const auto k1=rhs(x,a,b,u),k2=rhs(shifted(x,k1,h/2),a,b,u);
 const auto k3=rhs(shifted(x,k2,h/2),a,b,u),k4=rhs(shifted(x,k3,h),a,b,u);
 state out{};for(std::size_t i=0;i<4;++i)out[i]=x[i]+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;
 return out;
}
inline state initial(std::size_t c) {return {.1+.001*double(c),.2+.002*double(c),.8+.003*double(c),.05};}
inline state rollout(state x,double a=.8,double b=.7,int n=200) {
 // Endpoint is exactly 2.0; n must put the forcing discontinuity on a step boundary.
 if(n<=0 || n%8)throw std::invalid_argument("reference n must be positive and divisible by eight");
 const double dt=2.0/n;
 for(int i=0;i<n;++i) {const double u=i<(3*n)/8?1.0:0.0;x=step(x,a,b,u,dt);}
 return x;
}
inline std::array<double,2> parameter_gradient(const state& x) {
 // Since upstream x0,x1,x2 are parameter-independent, y is affine in a+b.
 const auto zero=rollout(x,0,0),unit=rollout(x,1,0);const double gain=unit[3]-zero[3];
 return {gain,gain};
}
inline double log_likelihood(double y,double mean,double sigma) {
 if(!(sigma>0))throw std::invalid_argument("positive sigma required");
 const double r=(y-mean)/sigma;return -.5*r*r-std::log(sigma)-.5*std::log(2*std::acos(-1.0));
}
inline std::array<double,2> normalize_log(double a,double b) {
 const double m=std::max(a,b),s=std::exp(a-m)+std::exp(b-m);return {std::exp(a-m)/s,std::exp(b-m)/s};
}
inline void require(bool v,const char* why) {if(!v)throw std::runtime_error(why);}
inline void near(double a,double b,double tol,const char* why) {require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tol,why);}
}
