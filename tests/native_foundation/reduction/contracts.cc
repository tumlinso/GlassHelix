#include <GlassHelix/reduction/contracts.hh>
using namespace glasshelix::reduction;

int main() {
  supplied_microstate a{1, 2, 7}, b{2, 1, 7};
  auto p = evaluate(a, .5, .5), q = evaluate(b, .5, .5);
  auto n = evaluate(a, .5, 1), m = evaluate(b, .5, 1);
  if (!p.closed || !q.closed || p.aggregate != q.aggregate || p.derivative != q.derivative ||
      p.reporter != q.reporter) return 1;
  if (n.closed || m.closed || n.aggregate != m.aggregate || n.derivative == m.derivative ||
      n.reporter != m.reporter) return 2;
  bool bad = false;
  try { (void)evaluate({1, 2, 3}, -1, 1); } catch (...) { bad = true; }

  error_report linear{.1, .1, .1, 99, .2, 1, true, "bounded linear", "study-17", "local"};
  error_report fail = linear, unresolved = linear;
  fail.certified = .3;
  unresolved.justified = false;
  // Equality passes, while a small empirical diagnostic cannot certify an unjustified bound.
  if (assess(linear, .2) != threshold_result::pass ||
      assess(fail, .2) != threshold_result::fail ||
      assess(unresolved, .2) != threshold_result::unresolved) return 3;

  bool negative = false, nonfinite = false, missing_provenance = false;
  linear.conversion = -1;
  try { (void)assess(linear, .2); } catch (...) { negative = true; }
  linear.conversion = .1;
  linear.integration = INFINITY;
  try { (void)assess(linear, .2); } catch (...) { nonfinite = true; }
  linear.integration = .1;
  linear.source = "";
  try { (void)assess(linear, .2); } catch (...) { missing_provenance = true; }
  return bad && negative && nonfinite && missing_provenance ? 0 : 4;
}
