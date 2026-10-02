#include <GlassHelix/interrogation/diagnostics.hh>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
bool check_nullspace(std::span<const double> matrix, std::size_t rows,
                     std::size_t columns, std::span<const double> scales,
                     std::size_t expected_rank) {
  const auto report = glasshelix::interrogation::rank_revealing_svd(
      matrix, rows, columns, scales, 1e-10, "rectangular");
  const auto nullity = columns - expected_rank;
  if (report.rank != expected_rank ||
      report.singular_values.size() != std::min(rows, columns) ||
      report.right_nullspace.size() != nullity * columns ||
      report.parameter_scales != std::vector<double>(scales.begin(), scales.end()))
    return false;

  double max_residual = 0.;
  for (std::size_t k = 0; k < nullity; ++k) {
    double residual_squared = 0.;
    for (std::size_t r = 0; r < rows; ++r) {
      double value = 0.;
      for (std::size_t c = 0; c < columns; ++c)
        value += matrix[r * columns + c] *
                 (report.right_nullspace[k * columns + c] / scales[c]);
      residual_squared += value * value;
    }
    max_residual = std::max(max_residual, std::sqrt(residual_squared));
    for (std::size_t j = 0; j <= k; ++j) {
      double dot = 0.;
      for (std::size_t c = 0; c < columns; ++c)
        dot += report.right_nullspace[k * columns + c] *
               report.right_nullspace[j * columns + c];
      if (std::abs(dot - (j == k ? 1. : 0.)) > 1e-10) return false;
    }
  }
  return std::isfinite(report.residual) && max_residual < 1e-10 &&
         std::abs(report.residual - max_residual) < 1e-10;
}
}

int main() {
  using glasshelix::interrogation::rank_revealing_svd;
  const std::array<double, 4> unresolved_matrix{1., 1., 2., 2.};
  const std::array<double, 4> resolved_matrix{1., 0., 0., 1.};
  const std::array<double, 2> unit_scales{1., 1.};
  const std::array<double, 2> nontrivial_scales{2., .5};
  auto unresolved = rank_revealing_svd(unresolved_matrix, 2, 2, unit_scales, 1e-10, "local");
  auto resolved = rank_revealing_svd(resolved_matrix, 2, 2, unit_scales, 1e-10, "control");
  auto scaled = rank_revealing_svd(unresolved_matrix, 2, 2, nontrivial_scales, 1e-10, "scaled");
  if (unresolved.rank != 1 || unresolved.right_nullspace.size() != 2 ||
      unresolved.local_domain != "local" || unresolved.residual > 1e-10 ||
      resolved.rank != 2 || !resolved.right_nullspace.empty() ||
      scaled.rank != 1 || scaled.residual > 1e-10 || scaled.parameter_scales !=
      std::vector<double>(nontrivial_scales.begin(), nontrivial_scales.end())) return 1;

  // The reported null vector is in scaled coordinates.  Transforming it back
  // to physical parameters must still annihilate the unscaled sensitivity.
  const auto& v = scaled.right_nullspace;
  const double d0 = v[0] / nontrivial_scales[0];
  const double d1 = v[1] / nontrivial_scales[1];
  const double physical_residual = std::hypot(d0 + d1, 2. * d0 + 2. * d1);
  if (physical_residual > 1e-10) return 2;

  // Wide matrices include null directions beyond the returned singular values.
  // Nonuniform scales exercise the documented A / scales coordinates.
  const std::array<double, 8> wide{1., 2., 3., 4., 2., -1., 1., 3.};
  const std::array<double, 8> wide_rank_one{1., 2., 3., 4., 2., 4., 6., 8.};
  const std::array<double, 4> wide_scales{2., .5, 3., 4.};
  const std::array<double, 12> tall{1., 2., 3., 2., -1., 1.,
                                     3., 1., 4., -1., 3., 2.};
  const std::array<double, 3> tall_scales{.5, 2., 3.};
  const std::array<double, 6> zero{};
  if (!check_nullspace(wide, 2, 4, wide_scales, 2) ||
      !check_nullspace(wide_rank_one, 2, 4, wide_scales, 1) ||
      !check_nullspace(tall, 4, 3, tall_scales, 2) ||
      !check_nullspace(zero, 2, 3, tall_scales, 0)) return 4;

  bool bad_shape = false, bad_threshold = false, bad_scale = false;
  const std::array<double, 1> bad_shape_scale{1.};
  try { (void)rank_revealing_svd(unresolved_matrix, 2, 2, bad_shape_scale, 1e-10, "x"); }
  catch (...) { bad_shape = true; }
  try { (void)rank_revealing_svd(unresolved_matrix, 2, 2, unit_scales,
                                 std::numeric_limits<double>::infinity(), "x"); }
  catch (...) { bad_threshold = true; }
  const std::array<double, 2> infinite_scale{1., std::numeric_limits<double>::infinity()};
  try { (void)rank_revealing_svd(unresolved_matrix, 2, 2, infinite_scale, 1e-10, "x"); }
  catch (...) { bad_scale = true; }
  return bad_shape && bad_threshold && bad_scale ? 0 : 3;
}
