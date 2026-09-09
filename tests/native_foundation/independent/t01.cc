#include "../fixtures/truth_known.hh"
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace gh_nf1_truth;
namespace {
int checks=0, rejected=0;
void require(bool value,const char* why) { ++checks; if(!value) throw std::runtime_error(why); }
bool close(double a,double b,double atol=2e-13,double rtol=2e-13) {
    return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=atol+rtol*std::abs(b);
}
void near(double a,double b,const char* why,double tol=2e-13) { require(close(a,b,tol,tol),why); }
void reject(double candidate,double oracle,const char* why) {
    require(!close(candidate,oracle,2e-9,2e-9),why); ++rejected;
}
// Adversarial adapter fixture deliberately uses a distinct slot order and symbolic lookup.
// Neither this slot layout nor its lookup enters the equations in truth_known.hh.
enum class Key { Reporter, Hidden, Partner, Driver };
using Row=std::pair<Key,double>;
double get(const std::array<Row,4>& rows,Key key) {
    for(const auto& [id,value]:rows) if(id==key) return value;
    throw std::runtime_error("missing identity");
}
double packed_reporter(const std::array<Row,4>& rows,Parameters p) {
    const auto d=get(rows,Key::Driver), h=get(rows,Key::Hidden);
    const auto q=get(rows,Key::Partner), y=get(rows,Key::Reporter);
    return p.a*(std::tanh(d)*q*h/(1+q*q)) + p.b*(std::tanh(d)*q*h/(1+q*q)) - .6*y;
}
void linear() {
    const State x{.3,0,0,0};
    const double exact=linear_exact(x.driver,.7,1.,.4);
    // Analytic integrating-factor expression against explicit stage integration.
    State numerical=x;
    for(int i=0;i<40;++i) numerical=rk4(numerical,{.8,.7},1.,.01);
    near(numerical.driver,exact,"linear RK4 versus analytic solution",2e-11);
    near(linear_exact(.2,0.,.5,2.),1.2,"zero-decay analytic limit");
    reject(x.driver+.4*(1.-.7*x.driver),exact,"Euler mislabeled as exact is detected");
    reject(linear_exact(x.driver,-.7,1.,.4),exact,"decay sign mutation is detected");
    const double e1=std::abs(rk4(x,{.8,.7},1.,.4).driver-exact);
    const auto half=rk4(rk4(x,{.8,.7},1.,.2),{.8,.7},1.,.2);
    require(e1/std::abs(half.driver-exact)>14,"fourth-order refinement must be visible");
}
void nonlinear_and_maps() {
    const Parameters p{.8,.7};
    for(int width: {0,1,16,33}) {
        std::vector<State> inputs;
        for(int i=0;i<width;++i) inputs.push_back({-.4+.07*i,.2+.03*i,.8+.02*i,.05-.001*i});
        int compared=0;
        for(const auto& s:inputs) {
            std::array<Row,4> rows{{{Key::Reporter,s.reporter},{Key::Hidden,s.hidden},
                                    {Key::Partner,s.partner},{Key::Driver,s.driver}}};
            const double expected=field(s,p,.3).reporter;
            // All 24 storage permutations retain identity and hence the same answer.
            std::sort(rows.begin(),rows.end(),[](const Row&a,const Row&b){return a.first<b.first;});
            do {
                near(packed_reporter(rows,p),expected,"independent identity mapping"); ++compared;
            } while(std::next_permutation(rows.begin(),rows.end(),[](const Row&a,const Row&b){return a.first<b.first;}));
        }
        require(compared==24*width,"empty and tail inventories are real");
    }
    const State s{.4,.2,.8,.05};
    const double correct=field(s,p,1.).reporter;
    reject((p.a+p.b)*std::tanh(s.partner)*s.driver/(1+s.driver*s.driver)*s.hidden-.6*s.reporter,
           correct,"role-swap mutation detected");
    reject(p.a*triad(s)-.6*s.reporter,correct,"dropped tied contribution detected");
    State contaminated=s;
    contaminated.driver+=.1*(1.-.7*s.driver);
    reject(field(contaminated,p,1.).reporter,correct,"in-place snapshot contamination detected");
    near(field(s,{1.1,.4},1.).reporter,correct,"a+b supplied exact identity");
    const double h=1e-5;
    const double fd=(field(s,{p.a+h,p.b},1.).reporter-field(s,{p.a-h,p.b},1.).reporter)/(2*h);
    near(fd,triad(s),"analytic parameter direction",1e-11);
    require(!close(std::numeric_limits<double>::quiet_NaN(),correct),"NaN never passes comparator");
}
void hidden_and_reduction() {
    const State x{.1,.2,.8,.05}, hidden{.1,.2,1.6,.05};
    near(observe(x),observe(hidden),"equal present observation");
    const auto a=trajectory(x,{.8,.7},200), b=trajectory(hidden,{.8,.7},200);
    require(std::abs(observe(a)-observe(b))>.01,"hidden state changes future");
    reject(observe(a),observe(b),"observation-only state merger detected");
    const auto fine=trajectory(x,{.8,.7},400), finer=trajectory(x,{.8,.7},800);
    const double coarse_error=std::abs(a.reporter-fine.reporter),fine_error=std::abs(fine.reporter-finer.reporter);
    require(coarse_error<1e-8 && coarse_error>8*fine_error,"nonlinear trajectory refinement");
    // Exact closed sum reduction and multiple fibers, independent full-system equations.
    for(double z: {-1.,0.,2.}) for(double split: {-.3,.2,1.1}) {
        const double first=split, second=z-split;
        const double full=(.4-.7*first)+(-.2-.7*second);
        near(full,reduced_sum_field(z,.7,.4,-.2),"closed reduction across fibers");
    }
    const double z=.9, first=.2, second=.7;
    const double full=(.4-.7*first)+(-.2-.7*second);
    reject((.4-.7*z),full,"dropped forcing in reduction detected");
    // Projection onto reporter is nonclosed: same R(x), distinct DR(x) f(x).
    reject(field(x,{.8,.7},1.).reporter,field(hidden,{.8,.7},1.).reporter,
           "nonclosed reporter projection detected");
    bool invalid=false;try { (void)trajectory(x,{.8,.7},201); } catch(const std::invalid_argument&) { invalid=true; }
    require(invalid,"unaligned forcing-grid input rejected");
}
}
int main() try {
    linear(); nonlinear_and_maps(); hidden_and_reduction();
    require(rejected==8,"all declared negative controls executed");
    std::cout<<"{\"task\":\"GH-NF1-T01\",\"evidence\":\"executed_host_test\",\"production_backend\":false,\"gpu\":false,\"checks\":"<<checks<<",\"rejected_variants\":"<<rejected<<"}\n";
    return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
