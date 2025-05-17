// -- Dictionary.hh --

#ifndef GLASSHELIX_DICTIONARY_HH
#define GLASSHELIX_DICTIONARY_HH

#include <cstddef>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <fstream>
#include <sstream>

template<typename V>
inline V from_string(const std::string& s) {
    V v;
    std::istringstream(s) >> v;
    return v;
}

template<>
inline std::string from_string<std::string>(const std::string& s) {
    return s;
}

template<typename T, typename U>
class Dictionary {
    template<typename X, typename Y>
    friend class Dictionary;

    static constexpr bool int_key = std::is_integral_v<U>;

    std::unordered_map<T, U> forward_;
    std::conditional_t<
            int_key,
            std::vector<const T*>,
            std::unordered_map<U, const T*>
    > reverse_;

public:
    Dictionary() = default;

    void reserve(std::size_t n) {
        forward_.reserve(n);
        reverse_.reserve(n);
    }
    void clear() noexcept {
        forward_.clear();
        reverse_.clear();
    }
    [[nodiscard]] std::size_t size() const noexcept { return forward_.size(); }

    bool contains(const T& t) const { return forward_.count(t); }
    bool contains(const U& u) const {
        if constexpr (int_key) return u < reverse_.size();
        else                  return reverse_.count(u);
    }

    U get(const T& t) const { return forward_.at(t); }
    const T& get(const U& u) const {
        if constexpr (int_key) return *reverse_[u];
        else                  return *reverse_.at(u);
    }

    U operator[](const T& t) const { return get(t); }
    const T& operator[](const U& u) const { return get(u); }

    void emplace(const T& t, const U& u) {
        auto [it, ok] = forward_.emplace(t, u);
        if (!ok) return;
        if constexpr (int_key) {
            reverse_.push_back(&it->first);
        } else {
            reverse_.emplace(u, &it->first);
        }
    }

    U generate(const T& t) {
        auto it = forward_.find(t);
        if (it != forward_.end()) return it->second;
        U id = static_cast<U>(forward_.size());
        auto [ins,_] = forward_.emplace(t, id);
        if constexpr (int_key) {
            reverse_.push_back(&ins->first);
        } else {
            reverse_.emplace(id, &ins->first);
        }
        return id;
    }

    void readFromFile(const std::string& filename, bool skip_header, unsigned long entries) {
        if (!forward_.empty()) throw std::runtime_error("Dictionary not empty");
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Could not open file " + filename);
        clear();
        if (entries) reserve(entries);

        std::string line;
        if (skip_header) std::getline(file, line);
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto pos = line.find(',');
            if (pos == std::string::npos) continue;
            T t = from_string<T>(line.substr(0,pos));
            U u = from_string<U>(line.substr(pos+1));
            emplace(t, u);
        }
    }

    using const_iterator = typename std::unordered_map<T,U>::const_iterator;

    /// Iterate from the first stored key→value pair…
    const_iterator begin() const noexcept { return forward_.begin(); }
    const_iterator end()   const noexcept { return forward_.end(); }

    template<typename V>
    Dictionary<V, U> compose(const Dictionary<V, T>& other) const {
        Dictionary<V,U> out;
        out.reserve(other.size());
        for (auto const& [v,t] : other.forward_)
            if (contains(t)) out.emplace(v, get(t));
        return out;
    }

    template<typename W>
    Dictionary<T,W> compose(const Dictionary<U, W>& other) const {
        Dictionary<T,W> out;
        out.reserve(size());
        for (auto const& [t,u] : forward_)
            if (other.contains(u)) out.emplace(t, other.get(u));
        return out;
    }

    std::vector<U> translate(const std::vector<T>& keys) const {
        std::vector<U> out;
        out.reserve(keys.size());
        for (auto const& k : keys) {
            out.push_back( get(k) );        // forward_ lookup
        }
        return out;
    }

    // translate methods
    {
        std::vector<T> translate(const std::vector<U>& keys) const {
            std::vector<T> out;
            out.reserve(keys.size());
            for (auto const& k : keys) {
                out.push_back( get(k) );        // reverse_ lookup
            }
            return out;
        }

        std::vector<U> translate(const T* keys, std::size_t count) const {
            std::vector<U> out;
            out.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                out.push_back(get(keys[i]));    // throws if keys[i] missing
            }
            return out;
        }

        std::vector<T> translate(const U* keys, std::size_t count) const {
            std::vector<T> out;
            out.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                out.push_back(get(keys[i]));    // throws if keys[i] missing
            }
            return out;
        }

        void translate(const T* keys, std::size_t count, U* out) const {
            for (std::size_t i = 0; i < count; ++i) {
                out[i] = get(keys[i]);
            }
        }

        void translate(const U* keys, std::size_t count, T* out) const {
            for (std::size_t i = 0; i < count; ++i) {
                out[i] = get(keys[i]);
            }
        }
    }
};

#endif  // GLASSHELIX_DICTIONARY_HH