#include <GlassHelix/system/assembly.hh>
#include <stdexcept>
namespace glasshelix::system {
assembled_mechanism::assembled_mechanism(std::shared_ptr<const core::system_definition> scientific,
 std::size_t index,compiled_implementation implementation)
 :scientific_(std::move(scientific)),index_(index),implementation_(std::move(implementation)) {
 if(!scientific_||index_>=scientific_->mechanisms().size())throw std::invalid_argument("scientific mechanism absent");
 const auto& m=scientific_mechanism();
 if(m.arguments.size()!=implementation_.input_roles.size()||m.outputs.size()!=implementation_.output_roles.size()||m.outputs.size()!=implementation_.output_effects.size())throw std::invalid_argument("ordered numerical roles differ from scientific arity");
 axes_.reserve(m.arguments.size()+m.outputs.size());
 for(auto i:m.arguments)axes_.push_back(scientific_->quantities()[i].axis);
 for(auto i:m.outputs)axes_.push_back(scientific_->quantities()[i].axis);
 for(std::size_t i=0;i<m.arguments.size();++i)inputs_.push_back({implementation_.input_roles[i],{&axes_[i],1},scientific_->quantities()[m.arguments[i]].extent,implementation_.numeric.state_storage});
 for(std::size_t i=0;i<m.outputs.size();++i)outputs_.push_back({{implementation_.output_roles[i],{&axes_[m.arguments.size()+i],1},scientific_->quantities()[m.outputs[i]].extent,implementation_.numeric.output_storage},implementation_.assembly_owner,implementation_.output_effects[i]});
 if(nf::validate_compiled_block(numerical_block())!=nf::status::success)throw std::invalid_argument("invalid Cellerator compiled block");
}
const core::mechanism& assembled_mechanism::scientific_mechanism()const{return scientific_->mechanisms()[index_];}
nf::compiled_block assembled_mechanism::numerical_block()const{
 nf::operation_contract c;c.definition=implementation_.numerical_definition;c.arguments=inputs_;c.outputs=outputs_;c.numeric=implementation_.numeric;
 if(implementation_.jvp)c.capabilities|=nf::jvp;
 if(implementation_.vjp)c.capabilities|=nf::vjp;
 if(implementation_.second)c.capabilities|=nf::second_direction;
 return {c,implementation_.effects,implementation_.forward,implementation_.jvp,implementation_.vjp,implementation_.second};
}
nf::status assembled_mechanism::bind_stage(nf::capability action,const void* prepared,std::uint64_t stage,std::uint64_t candidate,std::uint32_t binding,program::prepared_stage_v2& out)const{
 return nf::bind_compiled_stage(numerical_block(),action,prepared,stage,candidate,binding,out);
}
program::prepared_stage_v2 assembled_mechanism::bind_indexed_stage(
 const cellerator::compute::operation::indexed::prepared_evaluator_stage& stage,
 std::uint64_t stage_id,std::uint64_t candidate_id,std::uint32_t binding,
 std::uint64_t first_dependency,std::uint32_t dependency_count)const {
 if (!cellerator::compute::operation::indexed::valid_prepared_stage(stage))
   throw std::invalid_argument("invalid registered Cellerator block");
 return cellerator::compute::operation::indexed::make_prepared_stage(
   stage_id,candidate_id,stage,binding,first_dependency,dependency_count);
}
}
