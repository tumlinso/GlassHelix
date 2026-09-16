#pragma once

#include <GlassHelix/core/observation.hh>
#include <GlassHelix/observation/observation.hh>

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
  using predict_observation = std::function<std::vector<double>(const observation::engine_output&)>;

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

  void assimilate(const core::observation_record& observation,
                  const observation::gaussian_noise& likelihood,
                  const predict_observation& predictor, std::string evidence_id) {
    if (!observation.valid() || !likelihood.valid() || !predictor || evidence_id.empty()) {
      throw std::invalid_argument("invalid inference evidence");
    }
    if (propagated_.size() != candidates_.size()) {
      throw std::logic_error("each candidate must propagate through the common engine before assimilation");
    }
    std::vector<double> log_weights;
    log_weights.reserve(candidates_.size());
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t index = 0; index != candidates_.size(); ++index) {
      const auto prediction = predictor(propagated_[index]);
      const auto prior = candidates_[index].weight;
      const auto log_weight = std::log(prior) + likelihood.log_likelihood(observation, prediction);
      log_weights.push_back(log_weight);
      maximum = std::max(maximum, log_weight);
    }
    if (!std::isfinite(maximum)) throw std::domain_error("zero total likelihood");
    double total = 0;
    for (const auto value : log_weights) total += std::exp(value - maximum);
    if (!(total > 0) || !std::isfinite(total)) throw std::domain_error("zero total likelihood");
    for (std::size_t index = 0; index != candidates_.size(); ++index) {
      candidates_[index].weight = std::exp(log_weights[index] - maximum) / total;
    }
    evidence_history_.push_back(std::move(evidence_id));
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
      total += candidate_value.weight;
    }
    if (!(total > 0) || !std::isfinite(total)) throw std::invalid_argument("invalid candidate weights");
    for (auto& candidate_value : candidates_) candidate_value.weight /= total;
  }

  std::vector<candidate> candidates_;
  std::vector<observation::engine_output> propagated_;
  std::vector<std::string> evidence_history_;
};

}  // namespace glasshelix::inference
