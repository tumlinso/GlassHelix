#pragma once

#include <GlassHelix/core/observation.hh>
#include <GlassHelix/core/provenance.hh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace glasshelix::observation {

enum class map_kind { partial, linear, nonlinear };
enum class derivative_capability { unavailable, jvp_vjp };

// Numerical evolution remains owned by Cellerator.  This is the completed
// output of that common engine that an observation policy may read.
struct engine_output {
  std::vector<double> values;
  core::result_provenance provenance;
};

class observation_map {
 public:
  using nonlinear_forward = std::function<std::vector<double>(std::span<const double>)>;

  static observation_map partial(core::quantity output, std::size_t source_extent,
                                 std::vector<std::size_t> selected) {
    if (selected.empty()) throw std::invalid_argument("partial observation requires axes");
    for (const auto axis : selected) {
      if (axis >= source_extent) throw std::invalid_argument("partial observation axis out of range");
    }
    if (!valid_output(output, selected.size())) throw std::invalid_argument("invalid observation output");
    observation_map result{map_kind::partial, std::move(output), source_extent};
    result.selected_ = std::move(selected);
    return result;
  }

  static observation_map linear(core::quantity output, std::size_t source_extent,
                                std::vector<double> row_major_coefficients) {
    if (!valid_output(output, output.extent) || source_extent == 0 ||
        row_major_coefficients.size() != output.extent * source_extent) {
      throw std::invalid_argument("invalid linear observation");
    }
    for (const auto value : row_major_coefficients) {
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite linear coefficient");
    }
    observation_map result{map_kind::linear, std::move(output), source_extent};
    result.coefficients_ = std::move(row_major_coefficients);
    return result;
  }

  static observation_map nonlinear(core::quantity output, std::size_t source_extent,
                                   nonlinear_forward forward,
                                   derivative_capability derivatives = derivative_capability::unavailable) {
    if (!valid_output(output, output.extent) || source_extent == 0 || !forward) {
      throw std::invalid_argument("invalid nonlinear observation");
    }
    observation_map result{map_kind::nonlinear, std::move(output), source_extent};
    result.forward_ = std::move(forward);
    result.derivatives_ = derivatives;
    return result;
  }

  [[nodiscard]] map_kind kind() const noexcept { return kind_; }
  [[nodiscard]] derivative_capability derivatives() const noexcept { return derivatives_; }
  [[nodiscard]] const core::quantity& output() const noexcept { return output_; }

  [[nodiscard]] std::vector<double> evaluate(const engine_output& source) const {
    if (!source.provenance.valid()) throw std::invalid_argument("invalid common-engine provenance");
    if (source.values.size() != source_extent_) throw std::invalid_argument("common-engine output extent mismatch");
    for (const auto value : source.values) {
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite common-engine observation input");
    }
    std::vector<double> result;
    if (kind_ == map_kind::partial) {
      result.reserve(selected_.size());
      for (const auto axis : selected_) result.push_back(source.values.at(axis));
    } else if (kind_ == map_kind::linear) {
      result.assign(output_.extent, 0.0);
      for (std::size_t row = 0; row != output_.extent; ++row) {
        for (std::size_t col = 0; col != source_extent_; ++col) {
          result[row] += coefficients_[row * source_extent_ + col] * source.values[col];
        }
      }
    } else {
      result = forward_(source.values);
    }
    if (result.size() != output_.extent) throw std::invalid_argument("observation output extent mismatch");
    for (const auto value : result) {
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite observation output");
    }
    return result;
  }

 private:
  observation_map(map_kind kind, core::quantity output, std::size_t source_extent)
      : kind_(kind), output_(std::move(output)), source_extent_(source_extent) {}

  static bool valid_output(const core::quantity& output, std::size_t expected_extent) {
    return core::valid(output) && output.role == core::quantity_role::observation &&
           output.extent == expected_extent;
  }

  map_kind kind_;
  core::quantity output_;
  std::size_t source_extent_ = 0;
  std::vector<std::size_t> selected_;
  std::vector<double> coefficients_;
  nonlinear_forward forward_;
  derivative_capability derivatives_ = derivative_capability::unavailable;
};

struct gaussian_noise {
  double sigma = 0;
  std::uint64_t seed = 0;

  [[nodiscard]] bool valid() const noexcept { return std::isfinite(sigma) && sigma > 0; }

  [[nodiscard]] double log_likelihood(const core::observation_record& observation,
                                      std::span<const double> mean) const {
    if (!valid() || !observation.valid() || mean.size() != observation.values.size()) {
      throw std::invalid_argument("invalid Gaussian observation likelihood");
    }
    constexpr double log_two_pi = 1.8378770664093454836;
    double total = 0;
    for (std::size_t index = 0; index != mean.size(); ++index) {
      if (!observation.present[index]) continue;
      if (!std::isfinite(observation.values[index]) || !std::isfinite(mean[index])) {
        throw std::invalid_argument("nonfinite observed value or observation mean");
      }
      const auto normalized = (observation.values[index] - mean[index]) / sigma;
      total += -0.5 * normalized * normalized - std::log(sigma) - 0.5 * log_two_pi;
    }
    return total;
  }

  [[nodiscard]] core::observation_record sample(core::observation_record template_record,
                                                 std::span<const double> mean) const {
    if (!valid() || !template_record.valid() || mean.size() != template_record.values.size()) {
      throw std::invalid_argument("invalid Gaussian observation sample");
    }
    std::mt19937_64 generator{seed};
    std::normal_distribution<double> noise{0.0, sigma};
    for (std::size_t index = 0; index != mean.size(); ++index) {
      if (!template_record.present[index]) continue;
      if (!std::isfinite(mean[index])) throw std::invalid_argument("nonfinite observation mean");
      template_record.values[index] = mean[index] + noise(generator);
    }
    return template_record;
  }
};

}  // namespace glasshelix::observation
