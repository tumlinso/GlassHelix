#ifndef GLASSHELIX_BIDIRECTIONALMAP_HH
#define GLASSHELIX_BIDIRECTIONALMAP_HH

#include <unordered_map>

template<typename T, typename U>
class BidirectionalMap {
protected:
    std::unordered_map<T, U> forward{};
    std::unordered_map<U, T> reverse{};
public:
    BidirectionalMap() = default;
    void emplace(T t, U u) {
        forward.emplace(t, u);
        reverse.emplace(u, t);
    }
    U operator[](T t) const                     { return forward.at(t); }
    T operator[](U u) const                     { return reverse.at(u); }
    bool contains(T t) const                    { return forward.find(t) != forward.end(); }
    bool contains(U u) const                    { return reverse.find(u) != reverse.end(); }
    [[nodiscard]] unsigned long size() const    { return forward.size(); }
};

#endif //GLASSHELIX_BIDIRECTIONALMAP_HH
