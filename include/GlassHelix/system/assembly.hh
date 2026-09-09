#pragma once
#include <GlassHelix/core/system.hh>
#include <Cellerator/compute/operation/native_foundation_contract.hh>
namespace glasshelix::system {
namespace nf = cellerator::compute::operation::nf1;
namespace program = cellerator::execution::program;
struct compiled_implementation {
 nf::identity numerical_definition{}, assembly_owner{};
 std::vector<nf::identity> input_roles, output_roles;
 std::vector<cellerator::execution::output_effect_contract> output_effects;
 cellerator::compute::operation::v2::numerical_policy numeric{};
 nf::dependency_effects effects{};
 program::stage_launch_v2 forward=nullptr, jvp=nullptr, vjp=nullptr, second=nullptr;
};
// Owns signature backing memory; numerical launch state remains the CE/caller owner.
class assembled_mechanism {
 std::shared_ptr<const core::system_definition> scientific_;
 std::size_t index_;
 compiled_implementation implementation_;
 std::vector<cellerator::execution::persistent_axis_identity> axes_;
 std::vector<nf::operand_signature> inputs_;
 std::vector<nf::output_signature> outputs_;
public:
 assembled_mechanism(std::shared_ptr<const core::system_definition>,std::size_t,compiled_implementation);
 assembled_mechanism(const assembled_mechanism&)=delete;
 assembled_mechanism& operator=(const assembled_mechanism&)=delete;
 const core::mechanism& scientific_mechanism()const;
 nf::compiled_block numerical_block()const;
 nf::status bind_stage(nf::capability,const void*,std::uint64_t,std::uint64_t,std::uint32_t,program::prepared_stage_v2&)const;
};
using immutable_mechanism=std::shared_ptr<const assembled_mechanism>;
}
