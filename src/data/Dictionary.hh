#ifndef GLASSHELIX_DICTIONARY_HH
#define GLASSHELIX_DICTIONARY_HH

// Dictionary.hh

#include <cstddef>
#include <type_traits>
#include <vector>
#include <unordered_map>

template<typename T, typename U>
class Dictionary {
    static constexpr bool is_int = std::is_integral_v<U>;

    std::unordered_map<T, U> forward_;
    using ReverseType = std::conditional_t<
            is_int,
            std::vector<const T*>,
            std::unordered_map<U, const T*>
    >;
    ReverseType reverse_;

public:
    Dictionary() = default;

    // these are small—declare them inline:
    void reserve(std::size_t n);
    std::size_t size() const noexcept;
    bool contains(const T& t) const;
    bool contains(const U& u) const;
    U       get(const T& t) const;
    const T& get(const U& u) const;
    U       operator[](const T& t) const;
    const T& operator[](const U& u) const;

    // heavier, defined out-of-line:
    void emplace(const T& t, const U& u);
    U     generate(const T& t);
};

#include "Dictionary.inl"

#endif // GLASSHELIX_DICTIONARY_HH