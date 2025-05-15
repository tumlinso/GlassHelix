// Dictionary.cc

#include "Dictionary.hh"

template<typename T, typename U>
void Dictionary<T,U>::emplace(const T& t, const U& u) {
    auto [ins, ok] = forward_.emplace(t, u);
    if (!ok) return;
    const T* ptr = &ins->first;
    if constexpr (is_int) {
        reverse_.push_back(ptr);
    } else {
        reverse_.emplace(u, ptr);
    }
}

template<typename T, typename U>
U Dictionary<T,U>::generate(const T& t) {
    auto it = forward_.find(t);
    if (it != forward_.end()) return it->second;
    U id = static_cast<U>(forward_.size());
    auto [ins,_] = forward_.emplace(t, id);
    const T* ptr = &ins->first;
    if constexpr (is_int) {
        reverse_.push_back(ptr);
    } else {
        reverse_.emplace(id, ptr);
    }
    return id;
}
