#pragma once
#include <GlassHelix/core/quantity.hh>
#include <memory>
#include <stdexcept>
#include <unordered_set>
namespace glasshelix::core {
struct mechanism {
  std::string scientific_id;
  std::vector<std::size_t> arguments;
  std::vector<std::string> argument_roles;
  std::vector<std::size_t> outputs;
  bool repeated_arguments = false;
};
class system_definition {
  std::string id_;
  std::vector<quantity> quantities_;
  std::vector<mechanism> mechanisms_;
public:
  system_definition(std::string id, std::vector<quantity> q, std::vector<mechanism> m)
    : id_(std::move(id)), quantities_(std::move(q)), mechanisms_(std::move(m)) {
    if(id_.empty()) throw std::invalid_argument("scientific system id required");
    for(const auto& x:quantities_) if(!valid(x)) throw std::invalid_argument("invalid quantity");
    std::unordered_set<std::string> names;
    for(const auto& x:mechanisms_) {
      if(x.scientific_id.empty() || !names.insert(x.scientific_id).second || x.arguments.size()!=x.argument_roles.size() || x.outputs.empty()) throw std::invalid_argument("invalid mechanism");
      std::unordered_set<std::size_t> seen;
      for(auto a:x.arguments) if(a>=quantities_.size() || (!seen.insert(a).second && !x.repeated_arguments)) throw std::invalid_argument("invalid argument");
      for(auto a:x.outputs) if(a>=quantities_.size()) throw std::invalid_argument("invalid output");
      for(const auto& r:x.argument_roles) if(r.empty()) throw std::invalid_argument("argument role required");
    }
  }
  const auto& id()const noexcept{return id_;}
  const auto& quantities()const noexcept{return quantities_;}
  const auto& mechanisms()const noexcept{return mechanisms_;}
};
// Scientific definition ownership does not own or identify a numerical kernel.
struct system_instance {
  std::shared_ptr<const system_definition> definition;
  std::vector<double> state;
  std::vector<double> parameters;
  identity::value_generation state_generation{};
  identity::value_generation parameter_generation{};
};
struct realization_link { std::string mechanism_id; std::vector<std::string> numerical_stage_ids; };
}
