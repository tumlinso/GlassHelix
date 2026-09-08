#pragma once
#ifndef GH_NF1_DECLARATION_CHECK_ONLY
#error "Planning declarations only. Implement the actual library; never install or link this file."
#endif
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace glasshelix {
enum class backend { cpu, cuda };
enum class support_strategy { masked, compacted };
struct field {std::uint64_t id;};
struct parameter {std::uint64_t id;};
struct expression {std::uint64_t id;};
struct operation {std::uint64_t id;};
expression operator+(expression,expression);expression operator-(expression,expression);expression operator*(expression,expression);expression operator/(expression,expression);
expression operator*(double,expression);expression operator+(double,expression);expression operator+(expression,double);
expression tanh(expression);expression square(expression);
struct structure_definition {std::uint64_t id,epoch;};
struct observation {std::uint64_t id;};
struct statistics {std::uint64_t structure_preparations,hot_allocations,canonicalizations,launches;};
struct completion {void wait() const;};
struct simulation {completion ready;std::vector<double> download(observation)const;};
struct differential {completion ready;std::vector<double> download()const;};
struct null_result {std::size_t numerical_rank;std::vector<double> singular_values;std::vector<std::vector<double>> null_directions;double residual;bool local_only;};
struct candidate_state {std::vector<double> coordinates;double weight;};
struct inference_state {std::vector<candidate_state> candidates;std::vector<double> weights()const;};
struct inference_update {completion ready;};
struct session {
 struct impl;std::unique_ptr<impl> p;
 explicit session(backend,int=0);~session();session(session&&)noexcept;session&operator=(session&&)noexcept;
 session(const session&)=delete;backend actual_backend()const;std::string explain()const;
};
struct instance {
 struct impl;std::unique_ptr<impl> p;
 ~instance();instance(instance&&)noexcept;instance&operator=(instance&&)noexcept;instance(const instance&)=delete;
 void set_initial(std::span<const double> row_major,std::size_t batch);
 void publish_parameters(std::span<const double> values);
};
struct system_builder {
 field state(std::string_view);parameter parameter_value(std::string_view);
 expression read(field);expression read(parameter);expression forcing(std::string_view);expression argument(std::size_t);
 operation nary(std::string_view,std::span<const field>,expression);
 expression result(operation);void derivative(field,expression);observation observe(std::string_view,expression);
 structure_definition finish();
};
struct prepared_system {
 struct impl;std::unique_ptr<impl> p;
 prepared_system(session&,structure_definition);~prepared_system();prepared_system(prepared_system&&)noexcept;
 instance instantiate(std::span<const double> parameters);
 statistics inspect()const;
 simulation rk4(instance&,double begin,double end,double dt,std::span<const std::array<double,2>> piecewise_input);
 differential parameter_jvp(instance&,observation,std::span<const double> direction,double begin,double end,double dt,std::span<const std::array<double,2>>);
 differential parameter_vjp(instance&,observation,std::span<const double> output_cotangent,double begin,double end,double dt,std::span<const std::array<double,2>>);
 null_result local_parameter_nulls(instance&,observation,std::span<const double> parameter_scales,double begin,double end,double dt,std::span<const std::array<double,2>>);
 inference_update assimilate(inference_state&,observation,double measurement,double sigma)const;
 inference_update propagate(inference_state&,std::span<const double> parameters,double begin,double end,double dt,std::span<const std::array<double,2>>);
 void clear_activity(operation);
 void set_activity(operation,std::span<const std::uint8_t>,support_strategy);
};
}
