#pragma once

#include <GlassHelix/core/observation.hh>
#include <GlassHelix/observation/observation.hh>
#include <Cellerator/execution/program/program_v2.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace glasshelix::inference {

// A supplied candidate is a joint alternative.  It is never synthesized by
// marginalizing individual reporter coordinates or mechanism alternatives.
struct candidate {
  std::string hypothesis_id;
  std::string mechanism_id;
  std::vector<double> joint_state;
  double weight = 0;
};

class finite_candidates {
 public:
  using propagate_with_common_engine = std::function<observation::engine_output(const candidate&)>;

  explicit finite_candidates(std::vector<candidate> supplied) : candidates_(std::move(supplied)) {
    normalize_or_throw();
  }

  [[nodiscard]] const std::vector<candidate>& values() const noexcept { return candidates_; }
  [[nodiscard]] const std::vector<std::string>& evidence_history() const noexcept { return evidence_history_; }

  // Calls the supplied Cellerator-backed propagation route once per supplied
  // joint candidate.  GlassHelix does not implement a state-transition engine.
  void propagate(const propagate_with_common_engine& common_engine) {
    if (!common_engine) throw std::invalid_argument("common propagation engine required");
    propagated_.clear();
    propagated_.reserve(candidates_.size());
    for (const auto& candidate_value : candidates_) {
      const auto result = common_engine(candidate_value);
      if (!result.provenance.valid()) throw std::invalid_argument("common-engine propagation lacks provenance");
      propagated_.push_back(result);
    }
  }

  // CE owns execution. The binding provider owns candidate-resident state and
  // parameters; the collector is the explicit synchronized host egress.
  using ce_binding_provider = std::function<std::vector<cellerator::execution::program::launch_binding_v2>(const candidate&)>;
  using ce_result_collector = std::function<observation::engine_output(const candidate&)>;
  void propagate_ce(const cellerator::execution::program::prepared_program_v2& program,
                    const ce_binding_provider& bindings, const ce_result_collector& collect,
                    void* caller_stream) {
    if (!program.stages || !program.stage_count || !bindings || !collect)
      throw std::invalid_argument("CE propagation contract required");
    propagated_.clear(); propagated_.reserve(candidates_.size());
    for (const auto& candidate_value : candidates_) {
      auto candidate_bindings = bindings(candidate_value);
      const auto status = cellerator::execution::program::execute_prepared_program_v2(
          program, candidate_bindings.data(), candidate_bindings.size(), caller_stream);
      if (status != cellerator::execution::program::program_status::success)
        throw std::runtime_error("CE candidate propagation failed");
      auto result = collect(candidate_value);
      if (!result.provenance.valid()) throw std::invalid_argument("CE propagation lacks provenance");
      propagated_.push_back(std::move(result));
    }
  }

  void assimilate(const core::observation_record& observation,
                  const observation::gaussian_noise& likelihood,
                  const observation::observation_map& map, std::string evidence_id) {
    if (!observation.valid() || !likelihood.valid() || evidence_id.empty()) {
      throw std::invalid_argument("invalid inference evidence");
    }
    if (!core::interchangeable(observation.measured, map.output())) {
      throw std::invalid_argument("prediction observation identity mismatch");
    }
    if (std::find(evidence_history_.begin(), evidence_history_.end(), evidence_id) != evidence_history_.end()) {
      throw std::invalid_argument("duplicate evidence id");
    }
    if (propagated_.size() != candidates_.size()) {
      throw std::logic_error("each candidate must propagate through the common engine before assimilation");
    }
    std::vector<double> log_weights;
    log_weights.reserve(candidates_.size());
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index != candidates_.size(); ++index) {
      const auto prediction = map.evaluate(propagated_[index]);
      const auto log_weight = log_weights_[index] + likelihood.log_likelihood(observation, prediction);
      log_weights.push_back(log_weight);
      maximum = std::max(maximum, log_weight);
    }
    if (!std::isfinite(maximum)) throw std::domain_error("zero total likelihood");
    double total = 0;
    for (const auto value : log_weights) total += std::exp(value - maximum);
    if (!(total > 0) || !std::isfinite(total)) throw std::domain_error("zero total likelihood");
    const auto log_total = std::log(total);
    for (auto& value : log_weights) value = (value - maximum) - log_total;
    // Allocate evidence storage before committing the posterior. Failed updates
    // leave both the displayed weights and the authoritative log weights intact.
    evidence_history_.push_back(std::move(evidence_id));
    log_weights_.swap(log_weights);
    for (std::size_t index = 0; index != candidates_.size(); ++index) {
      candidates_[index].weight = std::exp(log_weights_[index]);
    }
  }

 private:
  void normalize_or_throw() {
    if (candidates_.empty()) throw std::invalid_argument("at least one supplied candidate required");
    double total = 0;
    for (const auto& candidate_value : candidates_) {
      if (candidate_value.hypothesis_id.empty() || candidate_value.mechanism_id.empty() ||
          candidate_value.joint_state.empty() || !std::isfinite(candidate_value.weight) ||
          candidate_value.weight < 0) {
        throw std::invalid_argument("invalid supplied candidate");
      }
      for (const auto coordinate : candidate_value.joint_state) {
        if (!std::isfinite(coordinate)) throw std::invalid_argument("nonfinite candidate joint state");
      }
      total += candidate_value.weight;
    }
    if (!(total > 0) || !std::isfinite(total)) throw std::invalid_argument("invalid candidate weights");
    log_weights_.reserve(candidates_.size());
    for (auto& candidate_value : candidates_) {
      // Capture the supplied prior before normalizing its display value: even
      // initial normalization can underflow a positive prior to zero.
      log_weights_.push_back(std::log(candidate_value.weight) - std::log(total));
      candidate_value.weight /= total;
    }
  }

  std::vector<candidate> candidates_;
  // Ordinary weights are a display projection and may underflow. Only an
  // explicitly zero supplied prior starts with a negative infinite log weight.
  std::vector<double> log_weights_;
  std::vector<observation::engine_output> propagated_;
  std::vector<std::string> evidence_history_;
};

}  // namespace glasshelix::inference
