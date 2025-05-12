export module BidirectionalMap;

import <unordered_map>;

export
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


// explicit use case for ULL id mapping
export
template <typename T>
class IDMap : virtual public BidirectionalMap<T, unsigned long> {
public:
    explicit IDMap() : BidirectionalMap<T, unsigned long>() {}

    unsigned long generate(T t) {
        if (this->contains(t)) {
            return this->forward[t];
        } else {
            unsigned long id = this->size();
            this->emplace(t, id);
            return id;
        }
    }
};
