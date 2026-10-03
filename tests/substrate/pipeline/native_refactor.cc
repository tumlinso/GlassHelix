// Actual fitted Sylvester generator, supplied permutation, CE epoch publication.
#include <Cellerator/math/adaptive/reuse.hh>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
namespace ad = cellerator::math::adaptive;
void require(bool ok) { if (!ok) throw std::runtime_error("native rewrite rejected"); }
int main() {
  double left[4], right[4];
  ad::linear_snapshot old;
  for (auto& v:left) require(bool(std::cin>>v) && std::isfinite(v));
  for (auto& v:right) require(bool(std::cin>>v) && std::isfinite(v));
  old.values.resize(4);
  for (auto& v:old.values) require(bool(std::cin>>v) && std::isfinite(v));
  old.generations={{1,1},{1},{1},{1},{1}};
  old.law.assign(16,0); old.readout.assign(16,0); old.outputs=4;
  old.first_moment.assign(4,1); old.second_moment.assign(4,2); old.optimizer_steps.assign(4,3);
  for (int i=0;i<4;++i) {
    old.coordinates.push_back({{1,1},{2,1},static_cast<unsigned long>(i),1});
    old.readout[4*i+i]=1;
    for (int j=0;j<4;++j) {
      // row-major vec(LX+XR): state coordinates stay distinct from generators.
      if(i%2==j%2) old.law[4*i+j]+=left[2*(i/2)+j/2];
      if(i/2==j/2) old.law[4*i+j]+=right[2*(j%2)+i%2];
    }
  }
  ad::publication owner(old);
  ad::supplied_rewrite proposal; proposal.candidate=old;
  auto& next=proposal.candidate;
  ++next.generations.epoch.value; ++next.generations.values.value; ++next.generations.parameters.value;
  proposal.forward.assign(16,0); proposal.backward.assign(16,0); proposal.optimizer_from.assign(4,-1);
  for(int i=0;i<4;++i) {
    ++next.coordinates[i].incarnation;
    next.values[i]=old.values[3-i];
    for(int j=0;j<4;++j) {
      next.law[4*i+j]=old.law[4*(3-i)+3-j];
      next.readout[4*i+j]=old.readout[4*i+3-j];
      proposal.forward[4*i+j]=proposal.backward[4*i+j]=double(i==3-j);
    }
  }
  { auto lease=owner.acquire_tape(); bool refused=false;
    try { owner.publish(proposal); } catch(const std::exception&) { refused=true; }
    require(refused && owner.snapshot().generations.epoch.value==1);
  }
  auto invalid=proposal; invalid.candidate.readout[0]+=1;
  bool refused=false; try { owner.publish(invalid); } catch(const std::exception&) { refused=true; }
  require(refused && owner.snapshot().generations.epoch.value==1);
  auto report=owner.publish(proposal);
  require(report.inverse_residual==0 && report.dynamics_residual==0 && report.readout_residual==0);
  require(owner.snapshot().first_moment==std::vector<double>(4,0));
  require(owner.snapshot().optimizer_steps==std::vector<std::uint64_t>(4,0));
  // Actual multistep numerical evolution and unchanged declared observation.
  auto x=old.values, q=owner.snapshot().values;
  double discrepancy=0;
  for(int step=0;step<6;++step) {
    std::vector<double> dx(4,0),dq(4,0);
    for(int i=0;i<4;++i) {
      double readout=0;
      for(int j=0;j<4;++j) {
        readout+=next.readout[4*i+j]*q[j];
        dx[i]+=old.law[4*i+j]*x[j]; dq[i]+=next.law[4*i+j]*q[j];
      }
      discrepancy=std::max(discrepancy,std::abs(x[i]-readout));
    }
    for(int i=0;i<4;++i) { x[i]+=.01*dx[i]; q[i]+=.01*dq[i]; }
  }
  require(discrepancy<1e-12);
  std::cout<<std::setprecision(17)<<"{\"owner\":\"CE adaptive publication\",\"kind\":\"exact_linear\","
    "\"old_epoch\":1,\"new_epoch\":2,\"reader_drain_checked\":true,"
    "\"invalid_readout_rejected\":true,\"optimizer_moments_reset\":true,"
    "\"trajectory_discrepancy\":"<<discrepancy<<",\"scope\":\"fitted autonomous Sylvester generator; Euler witness\"}\n";
}
