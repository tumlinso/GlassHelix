#pragma once

#include <GlassHelix/experiment/record.hh>
#include <GlassHelix/core/system.hh>
#include <algorithm>
#include <unordered_map>

namespace glasshelix::experiment {

struct measurement {
  core::observation_record observation;
  // Declared assay detection support is distinct from observation missingness.
  std::vector<bool> detection_support;
  std::string evidence_id;

  void validate() const {
    if (!observation.valid() || evidence_id.empty() ||
        detection_support.size() != observation.values.size())
      throw std::invalid_argument("measurement requires observation, detection and evidence identity");
    for (std::size_t i = 0; i < observation.values.size(); ++i)
      if (observation.present[i] && !std::isfinite(observation.values[i]))
        throw std::invalid_argument("observed value must be finite");
  }
};

enum class interpretation_scope { supplied_alternatives, globally_identified_mechanism };

struct prediction {
  std::string hypothesis_id;
  core::quantity observable;
  observation::engine_output output;
};

struct scientific_result {
  record receipt;
  std::vector<models::joint_candidate> whole_hypotheses;
  std::vector<std::string> conditioning_evidence_ids;
  std::vector<prediction> predictions;
  interpretation_scope scope = interpretation_scope::supplied_alternatives;
};

// Scientific composition only: execution and parameter storage retain CE and
// framework owners. Quantity roles remain those of the immutable typed system.
class specification {
  std::shared_ptr<const core::system_definition> system_;
  std::unordered_map<std::string, models::joint_candidate> whole_hypotheses_;
  std::size_t state_extent_ = 0;

  void validate_candidates(const models::candidate_set& candidates) const {
    if (candidates.values().size() != whole_hypotheses_.size())
      throw std::invalid_argument("result must preserve the complete supplied hypothesis set");
    std::unordered_set<std::string> seen;
    for (const auto& c : candidates.values()) {
      const auto found = whole_hypotheses_.find(c.hypothesis_id);
      if (found == whole_hypotheses_.end() || found->second.mechanism_id != c.mechanism_id ||
          found->second.joint_state != c.joint_state ||
          c.joint_state.size() != state_extent_ || !seen.insert(c.hypothesis_id).second)
        throw std::invalid_argument("whole hypothesis identity, mechanism or state extent changed");
    }
  }

 public:
  specification(std::shared_ptr<const core::system_definition> system,
                const models::candidate_set& candidates) : system_(std::move(system)) {
    if (!system_) throw std::invalid_argument("typed scientific system required");
    for (const auto& q : system_->quantities())
      if (q.role == core::quantity_role::dynamical_state) {
        if (q.extent > std::numeric_limits<std::size_t>::max() - state_extent_)
          throw std::invalid_argument("joint state extent overflow");
        state_extent_ += q.extent;
      }
    for (const auto& c : candidates.values()) {
      const auto known = std::any_of(system_->mechanisms().begin(), system_->mechanisms().end(),
                                    [&](const auto& m) { return m.scientific_id == c.mechanism_id; });
      if (!known || !whole_hypotheses_.emplace(c.hypothesis_id, c).second)
        throw std::invalid_argument("unique whole hypotheses must name supplied scientific mechanisms");
    }
    validate_candidates(candidates);
  }

  const core::system_definition& system() const noexcept { return *system_; }

  void validate_measurement(const measurement& measured) const {
    measured.validate();
    if (!std::any_of(system_->quantities().begin(), system_->quantities().end(),
                     [&](const auto& q) { return core::interchangeable(q, measured.observation.measured); }))
      throw std::invalid_argument("measurement is outside the scientific observation domain");
  }

  scientific_result snapshot(record receipt, const models::candidate_set& conditioned,
                             std::vector<prediction> predictions,
                             interpretation_scope scope = interpretation_scope::supplied_alternatives) const {
    validate_candidates(conditioned);
    receipt.validate(conditioned);
    if (receipt.provenance.scientific_model_id != system_->id())
      throw std::invalid_argument("receipt scientific model differs from specification");
    if (scope != interpretation_scope::supplied_alternatives)
      throw std::invalid_argument("local supplied-candidate inference does not establish global identification");
    std::unordered_set<std::string> predicted;
    for (const auto& p : predictions) {
      if (!whole_hypotheses_.contains(p.hypothesis_id) || !predicted.insert(p.hypothesis_id).second ||
          p.output.values.size() != p.observable.extent || !p.output.provenance.valid() ||
          p.output.provenance.scientific_model_id != system_->id() ||
          !std::any_of(system_->quantities().begin(), system_->quantities().end(),
                       [&](const auto& q) { return core::interchangeable(q, p.observable) &&
                                                  q.role == core::quantity_role::observation; }))
        throw std::invalid_argument("prediction requires one declared observable and whole hypothesis provenance");
      for (const auto value : p.output.values)
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite prediction");
    }
    if (predicted.size() != whole_hypotheses_.size())
      throw std::invalid_argument("prediction required for every whole hypothesis");
    return {std::move(receipt), conditioned.values(), conditioned.evidence_history(),
            std::move(predictions), scope};
  }
};

}  // namespace glasshelix::experiment
