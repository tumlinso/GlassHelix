#ifndef GLASSHELIX_SPARSEMATRIX_HH
#define GLASSHELIX_SPARSEMATRIX_HH

#include <cstddef>
#include <unordered_map>
#include <utility>

template <typename T, typename U>
struct pairHash {
    std::size_t operator()(const std::pair<T, U> &pair) const {
        return std::hash<T>()(pair.first) ^ std::hash<U>()(pair.second); // xor std::hash results lib std
    }
};

template <typename T, typename U, typename V>
struct MapMatrix { // composition, no inheritance from std::unordered_map, that would work maybe?
    std::unordered_map<std::pair<T, U>, V, pairHash<T, U>> data;

    // for (T, U) -> V
    [[maybe_unused]] void emplace(const T &t, const U &u, const V &v) { data.emplace({t, u}, v); }

    [[maybe_unused]] void emplace(const T &t, const U &u, V &&v) { data.emplace({t, u}, std::move(v)); }

    [[maybe_unused]] V& at(const T &t, const U &u) { return data.at({t, u}); }
    [[maybe_unused]] void erase(const T &t, const U &u) { data.erase({t, u}); }

    // alias for std::pair<T, U> -> V
    void emplace(const std::pair<T, U> &pair, const V &v) { data.emplace(pair, v); }
    void emplace(const std::pair<T, U> &pair, V &&v) { data.emplace(pair, std::move(v)); }
    V& at(const std::pair<T, U> &pair) { return data.at(pair); }
    [[maybe_unused]] void erase(const std::pair<T, U> &pair) { data.erase(pair); }

    // for (T, U) -> V
    V& operator()(const T &t, const U &u) { return data[{t, u}]; }
    const V& operator()(const T &t, const U &u) const { return data.at({t, u}); }

    [[maybe_unused]] [[nodiscard]] bool contains(const T &t, const U &u) const { return data.find({t, u}) != data.end(); }

    // alias for std::pair<T, U> -> V
    V& operator()(const std::pair<T, U> &pair) { return data[pair]; }
    const V& operator()(const std::pair<T, U> &pair) const { return data.at(pair); }
    [[nodiscard]] bool contains(const std::pair<T, U> &pair) const { return data.find(pair) != data.end(); }

    [[nodiscard]] [[maybe_unused]] unsigned long size() const { return data.size(); }
};

#endif //GLASSHELIX_SPARSEMATRIX_HH
