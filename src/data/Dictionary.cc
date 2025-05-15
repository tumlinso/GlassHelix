// Dictionary.cc

#include "Dictionary.hh"

template<typename T, typename U>
void Dictionary<T,U>::emplace(const T& t, const U& u) {
    auto [ins, ok] = forward_.emplace(t, u);
    if (!ok) return;
    const T* ptr = &ins->first;
    if constexpr (int_key) {
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
    if constexpr (int_key) {
        reverse_.push_back(ptr);
    } else {
        reverse_.emplace(id, ptr);
    }
    return id;
}

template<typename V>
V from_string(const std::string& s) {
    V v;
    std::istringstream(s) >> v;
    return v;
}

template<>
inline std::string from_string<std::string>(const std::string& s) {
    return s;
}

template<typename T, typename U>
void Dictionary<T,U>::readFromFile(const std::string& filename, bool skip_header, unsigned long entries) {
    if (forward_.size() != 0) throw std::runtime_error("Dictionary is not empty");
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Could not open dictionary file: " + filename);
    this->clear();
    if (entries != 0) { this->reserve(entries); }
    std::string line;
    if (skip_header) std::getline(file, line);
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto comma = line.find(',');
        if (comma == std::string::npos) continue;
        std::string one = line.substr(0, comma);
        std::string two = line.substr(comma + 1);
        T t = from_string<T>(one);
        U u = from_string<U>(two);
        emplace(t, u);
    }
}

template<typename T, typename U>
template<typename V>
Dictionary<V, U> Dictionary<T,U>::compose(const Dictionary<V, T>& other) const {
    Dictionary<V, U> result;
    result.reserve(other.size());
    // for each V→T in `other`
    for (auto const& [v, t] : other.forward_) {
        // if we have T→U in `this`
        if (this->contains(t)) {
            U u = this->get(t);
            result.emplace(v, u);
        }
    }
    return result;
}

template<typename T, typename U>
template<typename W>
Dictionary<T, W> Dictionary<T,U>::compose(const Dictionary<U, W>& other) const {
    Dictionary<T, W> result;
    result.reserve(this->size());
    // for each T→U in `this`
    for (auto const& [t, u] : forward_) {
        // if we have U→W in `other`
        if (other.contains(u)) {
            W w = other.get(u);
            result.emplace(t, w);
        }
    }
    return result;
}
