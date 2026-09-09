#pragma once
#include <GlassHelix/core/request.hh>
#include <functional>
#include <stdexcept>
namespace glasshelix::core {
enum class execution_status { success, unsupported, invalid_request };
enum class extension_kind { deterministic_forward, stochastic_transition, history, jvp, vjp, second_direction };
// A capability is callable only with its admitted invocation, never a bool flag.
// Numerical kernels, allocation and execution remain supplied by Cellerator.
class extension_dispatch {
 struct entry { extension_kind kind; std::function<execution_status()> invoke; };
 std::vector<entry> entries_;
public:
 void register_implementation(extension_kind kind,std::function<execution_status()> call){
  if(!call)throw std::invalid_argument("empty implementation");
  for(const auto& e:entries_)if(e.kind==kind)throw std::invalid_argument("duplicate extension");
  entries_.push_back({kind,std::move(call)});
 }
 execution_status invoke(extension_kind kind)const{
  for(const auto& e:entries_)if(e.kind==kind)return e.invoke();
  return execution_status::unsupported;
 }
};
}
