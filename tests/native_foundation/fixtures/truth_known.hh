#pragma once
// Test-only FP64 equations. No production header, lowering map, or kernel dependency.
#include <array>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace gh_nf1_truth {
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559
              && std::numeric_limits<double>::digits == 53);
struct State { double driver, partner, hidden, reporter; };
struct Parameters { double a, b; };

inline double triad(const State& s) {
    return std::tanh(s.driver) * s.partner / (1.0 + s.partner*s.partner) * s.hidden;
}
inline State field(const State& s, Parameters p, double forcing) {
    return {forcing - 0.7*s.driver,
            0.4*s.driver - 0.5*s.partner,
            0.2 + 0.3*s.partner - 0.4*s.hidden,
            (p.a+p.b)*triad(s) - 0.6*s.reporter};
}
inline State shift(const State& x, const State& d, double h) {
    return {x.driver+h*d.driver, x.partner+h*d.partner,
            x.hidden+h*d.hidden, x.reporter+h*d.reporter};
}
inline State rk4(const State& x, Parameters p, double forcing, double dt) {
    const auto a=field(x,p,forcing);
    const auto b=field(shift(x,a,dt/2),p,forcing);
    const auto c=field(shift(x,b,dt/2),p,forcing);
    const auto d=field(shift(x,c,dt),p,forcing);
    return {x.driver+dt*(a.driver+2*b.driver+2*c.driver+d.driver)/6,
            x.partner+dt*(a.partner+2*b.partner+2*c.partner+d.partner)/6,
            x.hidden+dt*(a.hidden+2*b.hidden+2*c.hidden+d.hidden)/6,
            x.reporter+dt*(a.reporter+2*b.reporter+2*c.reporter+d.reporter)/6};
}
inline State trajectory(State x, Parameters p, int steps) {
    if (steps<=0 || steps%8) throw std::invalid_argument("steps must align the 0.75 forcing split");
    const double dt=2.0/steps;
    // Finish the preceding interval with its forcing, including its endpoint stage.
    for (int i=0;i<steps;++i) x=rk4(x,p,i<3*steps/8 ? 1.0 : 0.0,dt);
    return x;
}
// Independent exact solution for diagonal affine continuous-time dynamics.
inline double linear_exact(double x, double decay, double forcing, double time) {
    if (decay==0) return x+forcing*time;
    return x*std::exp(-decay*time) - forcing*std::expm1(-decay*time)/decay;
}
// Supplied closed reduction: x'=u-k*x, y'=v-k*y => z=x+y, z'=u+v-k*z.
inline double reduced_sum_field(double z, double k, double u, double v) {
    return u+v-k*z;
}
inline double observe(const State& x) { return x.reporter; }
} // namespace gh_nf1_truth
