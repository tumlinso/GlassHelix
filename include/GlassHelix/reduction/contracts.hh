#pragma once
#include <cmath>
#include <stdexcept>
namespace glasshelix::reduction { struct supplied_microstate {double left,right,reporter;}; struct reduction_result {double aggregate,derivative,reporter;bool closed;}; inline reduction_result evaluate(supplied_microstate x,double left_rate,double right_rate){if(!std::isfinite(x.left)||!std::isfinite(x.right)||!std::isfinite(x.reporter)||!std::isfinite(left_rate)||!std::isfinite(right_rate)||left_rate<0||right_rate<0)throw std::invalid_argument("invalid supplied reduction regime");return {x.left+x.right,left_rate*x.left+right_rate*x.right,x.reporter,left_rate==right_rate};} }
