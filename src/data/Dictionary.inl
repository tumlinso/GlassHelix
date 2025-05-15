#ifndef GLASSHELIX_DICTIONARY_INL
#define GLASSHELIX_DICTIONARY_INL

#ifndef GLASSHELIX_DICTIONARY_HH
#include "Dictionary.hh"
#endif //GLASSHELIX_DICTIONARY_HH

// Dictionary.inl

template<typename T, typename U>
inline void Dictionary<T,U>::reserve(std::size_t n) {
    forward_.reserve(n);
    reverse_.reserve(n);
}

template<typename T, typename U>
inline void Dictionary<T,U>::clear() noexcept {
    forward_.clear();
    reverse_.clear();
}

template<typename T, typename U>
inline std::size_t Dictionary<T,U>::size() const noexcept {
    return forward_.size();
}

template<typename T, typename U>
inline bool Dictionary<T,U>::contains(const T& t) const {
    return forward_.find(t) != forward_.end();
}

template<typename T, typename U>
inline bool Dictionary<T,U>::contains(const U& u) const {
    if constexpr (int_key) return u < reverse_.size();
    else                  return reverse_.find(u) != reverse_.end();
}

template<typename T, typename U>
inline U Dictionary<T,U>::get(const T& t) const {
    return forward_.at(t);
}

template<typename T, typename U>
inline const T& Dictionary<T,U>::get(const U& u) const {
    if constexpr (int_key) return *reverse_[u];
    else                  return *reverse_.at(u);
}

template<typename T, typename U>
inline U Dictionary<T,U>::operator[](const T& t) const {
    return get(t);
}

template<typename T, typename U>
inline const T& Dictionary<T,U>::operator[](const U& u) const {
    return get(u);
}

#endif //GLASSHELIX_DICTIONARY_INL
