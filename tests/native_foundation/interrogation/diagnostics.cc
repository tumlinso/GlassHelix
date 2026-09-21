#include <GlassHelix/interrogation/diagnostics.hh>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

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
