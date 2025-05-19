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

//-----------------------------------------------------------------------------
// Utility: convert string to value (supports arithmetic types and std::string)
//-----------------------------------------------------------------------------

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

//-----------------------------------------------------------------------------
// Dictionary maps between keys of type T and values of type U.
// For integral U it also maintains a fast index-based reverse lookup.
//-----------------------------------------------------------------------------

template<typename T, typename U>
class Dictionary {
    // Allow composing with other Dictionary instantiations
    template<typename X, typename Y>
    friend class Dictionary;

    // If U is integral, we use a vector for reverse mapping
    static constexpr bool int_key = std::is_integral_v<U>;

    // Forward mapping: T -> U
    std::unordered_map<T, U> forward_;

    // Reverse mapping: U -> T* (pointer to key in forward_)
    std::conditional_t<
            int_key,
            std::vector<const T*>,              // vector indexed by U
            std::unordered_map<U, const T*>     // unordered_map otherwise
    > reverse_;

public:
    //-------------------------------------------------------------------------
    // Constructors / modifiers
    //-------------------------------------------------------------------------

    Dictionary() = default;
    Dictionary(std::string filename,
               bool skip_header = false,
               unsigned long entries = 0)
    {
        readFromFile(filename, skip_header, entries);
    }

    /// Preallocate space for n entries in both mappings
    void reserve(std::size_t n) {
        forward_.reserve(n);
        reverse_.reserve(n);
    }

    /// Remove all entries without deallocating memory
    void clear() noexcept {
        forward_.clear();
        reverse_.clear();
    }

    /// Number of stored key-value pairs
    [[nodiscard]] std::size_t size() const noexcept {
        return forward_.size();
    }

    //-------------------------------------------------------------------------
    // Query / lookup
    //-------------------------------------------------------------------------

    /// Check if a key exists
    bool contains(const T& t) const {
        return forward_.count(t) > 0;
    }

    /// Check if a value exists (reverse lookup)
    bool contains(const U& u) const {
        if constexpr (int_key) {
            return u < reverse_.size();
        } else {
            return reverse_.count(u) > 0;
        }
    }

    /// Get U for key T (throws std::out_of_range if not present)
    U get(const T& t) const {
        return forward_.at(t);
    }

    /// Get T for key U (throws if not present)
    const T& get(const U& u) const {
        if constexpr (int_key) {
            return *reverse_[u];
        } else {
            return *reverse_.at(u);
        }
    }

    /// Convenient operator[] overloads
    U operator[](const T& t) const { return get(t); }
    const T& operator[](const U& u) const { return get(u); }

    //-------------------------------------------------------------------------
    // Insertion
    //-------------------------------------------------------------------------

    /// Insert a specific mapping; does nothing if key already present
    void emplace(const T& t, const U& u) {
        auto [it, ok] = forward_.emplace(t, u);
        if (!ok) return;
        if constexpr (int_key) {
            reverse_.push_back(&it->first);
        } else {
            reverse_.emplace(u, &it->first);
        }
    }

    /// Generate a new U for T if not present; return existing value otherwise
    U generate(const T& t) {
        auto it = forward_.find(t);
        if (it != forward_.end()) return it->second;
        U id = static_cast<U>(forward_.size());
        auto [ins, _] = forward_.emplace(t, id);
        if constexpr (int_key) {
            reverse_.push_back(&ins->first);
        } else {
            reverse_.emplace(id, &ins->first);
        }
        return id;
    }

    //-------------------------------------------------------------------------
    // File I/O
    //-------------------------------------------------------------------------

    /**
     * Read key,value pairs from CSV file. Format: key,val per line.
     * If skip_header is true, skip the first line.
     * Pre-reserves 'entries' slots if provided.
     * Throws if dictionary is not empty or file cannot be opened.
     */
    void readFromFile(const std::string& filename,
                      bool skip_header,
                      unsigned long entries)
    {
        if (!forward_.empty())
            throw std::runtime_error("Dictionary not empty");

        std::ifstream file(filename);
        if (!file)
            throw std::runtime_error("Could not open file " + filename);

        clear();
        if (entries) reserve(entries);

        std::string line;
        if (skip_header) std::getline(file, line);

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            auto pos = line.find(',');
            if (pos == std::string::npos) continue;

            T t = from_string<T>(line.substr(0, pos));
            U u = from_string<U>(line.substr(pos + 1));
            emplace(t, u);
        }
    }

    //-------------------------------------------------------------------------
    // Iteration
    //-------------------------------------------------------------------------

    using const_iterator = typename std::unordered_map<T,U>::const_iterator;

    const_iterator begin() const noexcept { return forward_.begin(); }
    const_iterator end()   const noexcept { return forward_.end();   }

    //-------------------------------------------------------------------------
    // Composition
    //-------------------------------------------------------------------------

    /// Compose: V -> T dictionary followed by this T -> U gives V -> U
    template<typename V>
    Dictionary<V, U> compose(const Dictionary<V, T>& other) const {
        Dictionary<V,U> out;
        out.reserve(other.size());
        for (auto const& [v,t] : other.forward_)
            if (contains(t))
                out.emplace(v, get(t));
        return out;
    }

    /// Compose: T -> U followed by U -> W gives T -> W
    template<typename W>
    Dictionary<T, W> compose(const Dictionary<U, W>& other) const {
        Dictionary<T,W> out;
        out.reserve(size());
        for (auto const& [t,u] : forward_)
            if (other.contains(u))
                out.emplace(t, other.get(u));
        return out;
    }

    //-------------------------------------------------------------------------
    // Bulk translation
    //-------------------------------------------------------------------------

    /**
     * Translate a vector of T keys into a vector of U values.
     * Throws if any key is missing.
     */
    std::vector<U> translate(const std::vector<T>& keys) const {
        std::vector<U> out;
        out.reserve(keys.size());
        for (auto const& k : keys) {
            out.push_back(get(k));
        }
        return out;
    }

    /**
     * Translate a vector of U keys into a vector of T values.
     */
    std::vector<T> translate(const std::vector<U>& keys) const {
        std::vector<T> out;
        out.reserve(keys.size());
        for (auto const& k : keys) {
            out.push_back(get(k));
        }
        return out;
    }

    /**
     * Translate a C-array of T keys into a vector of U values.
     */
    std::vector<U> translate(const T* keys, std::size_t count) const {
        std::vector<U> out;
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            out.push_back(get(keys[i]));
        }
        return out;
    }

    /**
     * Translate a C-array of U keys into a vector of T values.
     */
    std::vector<T> translate(const U* keys, std::size_t count) const {
        std::vector<T> out;
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            out.push_back(get(keys[i]));
        }
        return out;
    }

    /**
     * Fill a pre-allocated array of U from T keys.
     */
    void translate(const T* keys, std::size_t count, U* out) const {
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = get(keys[i]);
        }
    }

    /**
     * Fill a pre-allocated array of T from U keys.
     */
    void translate(const U* keys, std::size_t count, T* out) const {
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = get(keys[i]);
        }
    }
};

#endif  // GLASSHELIX_DICTIONARY_HH
