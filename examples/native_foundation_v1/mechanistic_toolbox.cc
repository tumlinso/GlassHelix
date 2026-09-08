// Prospective public consumer. All numerical calls must reach the real GlassHelix/Cellerator stack.
// The independent oracle below checks results; it is never an execution fallback.
#include <GlassHelix/foundation.hh>
#include "fixture.hh"
#include <iostream>
#include <string_view>
int main(int argc,char**argv) try {
 namespace gh=glasshelix;using nf1_reference::require;using nf1_reference::near;
 const bool cpu=argc==2 && std::string_view(argv[1])=="--cpu";
 if(argc>1&&!cpu)throw std::invalid_argument("usage: mechanistic_toolbox [--cpu]; default requires CUDA");
 gh::session session(cpu?gh::backend::cpu:gh::backend::cuda);
 require(session.actual_backend()==(cpu?gh::backend::cpu:gh::backend::cuda),"silent backend fallback forbidden");
 gh::system_builder builder;
 const auto x0=builder.state("activity-0"),x1=builder.state("activity-1"),x2=builder.state("hidden-activity"),y=builder.state("observed-output");
 const auto a=builder.parameter_value("a"),b=builder.parameter_value("b");
 const std::array inputs{x0,x1,x2};
 const auto r0=builder.argument(0),r1=builder.argument(1),r2=builder.argument(2);
 const auto triad=builder.nary("joint-triad",inputs,gh::tanh(r0)*(r1/(1.0+gh::square(r1)))*r2);
 builder.derivative(x0,builder.forcing("external-input")-.7*builder.read(x0));
 builder.derivative(x1,.4*builder.read(x0)-.5*builder.read(x1));
 builder.derivative(x2,.2+.3*builder.read(x1)-.4*builder.read(x2));
 builder.derivative(y,(builder.read(a)+builder.read(b))*builder.result(triad)-.6*builder.read(y));
 const auto measured=builder.observe("partial-readout",builder.read(y));
 const auto definition=builder.finish();gh::prepared_system program(session,definition);
 const std::array<double,2> p{.8,.7},q{1.1,.4};
 auto first=program.instantiate(p),second=program.instantiate(q);
 constexpr std::size_t batch=33;std::vector<double> initial;initial.reserve(4*batch);
 for(std::size_t c=0;c<batch;++c){const auto x=nf1_reference::initial(c);initial.insert(initial.end(),x.begin(),x.end());}
 first.set_initial(initial,batch);second.set_initial(initial,batch);
 // Values at 0.75 are right-continuous; integrator splits there and uses each interval's input.
 const std::array<std::array<double,2>,2> forcing{{{0.,1.},{.75,0.}}};
 const auto a_run=program.rk4(first,0,2,.01,forcing),b_run=program.rk4(second,0,2,.01,forcing);
 a_run.ready.wait();b_run.ready.wait();const auto ya=a_run.download(measured),yb=b_run.download(measured);
 require(ya.size()==batch && yb.size()==batch,"batch tail lost");
 for(std::size_t c=0;c<batch;++c){near(ya[c],nf1_reference::rollout(nf1_reference::initial(c))[3],3e-5,"independent oracle mismatch");near(ya[c],yb[c],3e-5,"same effective parameter sum disagrees");}
 // Derivatives are explicitly of the RK4 observation map from the reset initial state.
 first.set_initial(initial,batch);const std::array<double,2> direction{1.,-1.};
 const auto jvp=program.parameter_jvp(first,measured,direction,0,2,.01,forcing);jvp.ready.wait();
 for(double v:jvp.download())near(v,0,3e-5,"known local null direction was lost");
 first.set_initial(initial,batch);std::vector<double> cotangent(batch,1.);
 const auto vjp=program.parameter_vjp(first,measured,cotangent,0,2,.01,forcing);vjp.ready.wait();const auto gradient=vjp.download();
 require(gradient.size()==2,"wrong parameter axis");double gain=0;
 for(std::size_t c=0;c<batch;++c)gain+=nf1_reference::parameter_gradient(nf1_reference::initial(c))[0];
 near(gradient[0],gain,3e-4,"parameter VJP wrong");near(gradient[1],gain,3e-4,"parameter tie incorrectly imposed");
 first.set_initial(initial,batch);const std::array<double,2> scales{1.,1.};
 const auto nulls=program.local_parameter_nulls(first,measured,scales,0,2,.01,forcing);
 require(nulls.local_only && nulls.numerical_rank==1 && nulls.null_directions.size()==1,"local ambiguity result wrong");
 require(nulls.residual<1e-4,"null residual not qualified");
 require(program.inspect().structure_preparations==1,"shared definition duplicated structural preparation");
 // Parameter changes affect one instance only, despite shared immutable structure.
 first.publish_parameters(std::array<double,2>{.9,.7});first.set_initial(initial,batch);second.set_initial(initial,batch);
 const auto changed=program.rk4(first,0,2,.01,forcing),unchanged=program.rk4(second,0,2,.01,forcing);changed.ready.wait();unchanged.ready.wait();
 near(unchanged.download(measured)[0],yb[0],3e-5,"independent value planes contaminated");
 require(std::abs(changed.download(measured)[0]-ya[0])>1e-3,"updated parameters unused");
 first.publish_parameters(p);
 // Identical active support, two physical routes. No claim that compaction is faster.
 std::vector<std::uint8_t> active(batch,1);active.back()=0;
 program.set_activity(triad,active,gh::support_strategy::masked);first.set_initial(initial,batch);
 const auto masked=program.rk4(first,0,2,.01,forcing);masked.ready.wait();const auto ym=masked.download(measured);
 program.set_activity(triad,active,gh::support_strategy::compacted);first.set_initial(initial,batch);
 const auto compact=program.rk4(first,0,2,.01,forcing);compact.ready.wait();const auto yc=compact.download(measured);
 for(std::size_t i=0;i<batch;++i)near(ym[i],yc[i],3e-5,"support realizations disagree");
 program.clear_activity(triad);
 auto low=nf1_reference::initial(0),high=low;high[2]+=.8;
 gh::inference_state belief{{{{low.begin(),low.end()},.5},{{high.begin(),high.end()},.5}}};
 auto assimilation=program.assimilate(belief,measured,low[3],.005);assimilation.ready.wait();
 near(belief.weights()[0],.5,1e-6,"current observation fabricated hidden certainty");
 auto propagation=program.propagate(belief,p,0,2,.01,forcing);propagation.ready.wait();
 assimilation=program.assimilate(belief,measured,nf1_reference::rollout(low)[3],.005);assimilation.ready.wait();
 require(belief.weights()[0]>.99,"new evidence did not refine supplied alternatives");
 const auto stats=program.inspect();require(stats.structure_preparations>=1,"missing execution accounting");
 // Alternate physical support preparation may increment its own counter; original shared structure must not be copied per instance.
 std::cout<<"native toolbox passed; backend="<<(cpu?"cpu":"cuda")<<" batch="<<batch<<" local_rank="<<nulls.numerical_rank<<"\n"<<session.explain()<<'\n';
 return 0;
} catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
