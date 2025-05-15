#ifndef GLASSHELIX_DICTIONARY_HH
#define GLASSHELIX_DICTIONARY_HH

// Dictionary.hh

#include <cstddef>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

template<typename T, typename U>
class Dictionary {
    static constexpr bool is_int = std::is_integral_v<U>;

    std::unordered_map<T, U> forward_;
    std::conditional_t<
            is_int,
            std::vector<const T*>,
            std::unordered_map<U, const T*>
    > reverse_;

public:
    Dictionary() = default;

    void reserve(std::size_t n);
    void clear() noexcept;
    std::size_t size() const noexcept;
    bool contains(const T& t) const;
    bool contains(const U& u) const;
    U       get(const T& t) const;
    const T& get(const U& u) const;
    U       operator[](const T& t) const;
    const T& operator[](const U& u) const;

    void emplace(const T& t, const U& u);
    U     generate(const T& t);

    void readFromFile(const std::string& filename, bool skip_header = false, unsigned long entries = 0);
};

#include "Dictionary.inl"

#endif // GLASSHELIX_DICTIONARY_HH