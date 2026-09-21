#pragma once
#include <cmath>
#include <stdexcept>
namespace glasshelix::reduction { struct error_report { double numerical=0, empirical=0, justified_bound=0; bool certified=false; }; inline void validate(error_report x,double threshold){if(!std::isfinite(x.numerical)||!std::isfinite(x.empirical)||x.justified_bound<0||!std::isfinite(x.justified_bound))throw std::invalid_argument("invalid error provenance");if(x.certified&&x.justified_bound>threshold)throw std::domain_error("unresolved justified bound");} inline bool equal_decay_closed(double a,double b,double rate){return std::isfinite(a)&&std::isfinite(b)&&rate>=0;} inline bool preserves_reporter_correlation(double x,double y){return x==y;} }
