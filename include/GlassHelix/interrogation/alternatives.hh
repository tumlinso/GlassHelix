#pragma once
#include <stdexcept>
#include <string>
#include <vector>
namespace glasshelix::interrogation { struct alternative {std::string id,role;std::vector<double> shared,exclusive;}; inline bool aligned(const alternative&a,const alternative&b){return !a.role.empty()&&a.role==b.role&&a.shared==b.shared;} inline bool exclusive_branch(const alternative&a,const alternative&b){return aligned(a,b)&&a.exclusive!=b.exclusive;} }
