#ifndef GLASSHELIX_SEQUENCE_HH
#define GLASSHELIX_SEQUENCE_HH

/**
 * Sequence.hh
 *
 * Defines the Sequence class for representing nucleotide sequences.
 * Sequences store bases compactly in packed blocks, supporting efficient access and modification.
 * Supports subsequence features with optional parent overlay.
 */

//--- Includes ---
#include <cstddef>
#include <cassert>
#include <string>
#include <cstdlib>   // for aligned_alloc, free
#include <new>       // for std::bad_alloc

#include "src/memory/Feature.hh"

namespace glasshelix {

// Allow customizing the underlying memory type (e.g. SIMD types)
template<typename Word = unsigned long>
class Sequence : public Feature {
public:
    /// Underlying memory type and compile-time constants
    static inline constexpr size_t BaseLengthBits = 4;
    static inline constexpr size_t BitsPerWord = sizeof(Word) * 8;
    static inline constexpr size_t BasesPerWord = BitsPerWord / BaseLengthBits;

    // BasesPerWord must be power of two for fast bit ops
    static_assert((BasesPerWord & (BasesPerWord - 1)) == 0,
                  "BasesPerWord must be a power of two");
    // compute log2(BasesPerWord) at compile time
    template<size_t N>
    struct Log2 { static constexpr size_t value = 1 + Log2<N/2>::value; };
    template<> struct Log2<1> { static constexpr size_t value = 0; };

    static inline constexpr size_t WordShift = Log2<BasesPerWord>::value;
    static inline constexpr size_t WordMask  = BasesPerWord - 1;
    static inline constexpr size_t BaseMask   = (1ull << BaseLengthBits) - 1;

//--- Base encoding ---
typedef enum {
    A = 0b0001,
    C = 0b0010,
    G = 0b0100,
    T = 0b1000,
    N = 0b0000
} Base;

/**
 * Sequence: stores DNA bases in packed blocks with optional parent overlay.
 * Supports base access, mutation, and subsequence operations.
 */
    friend class Feature;
public:
    /// Construct Sequence from a Feature and optional parent Sequence
    explicit Sequence(Feature feature, Sequence *parent = nullptr);
    /// Construct Sequence from a string representation of bases
    explicit Sequence(const std::string &sequenceString);
    /// Destructor releases allocated data if no parent
    ~Sequence() { if (!parent) std::free(data); }

    /// Get base at position (const)
    inline Base operator() (size_t pos) const;
    /// Set base at position
    inline void operator() (size_t pos, Base base);

    /// Get subsequence defined by feature
    inline Sequence operator[] (Feature feature);
    /// Set subsequence defined by feature to given sequence
    inline void operator[] (Feature feature, Sequence sequence);

    /// Convert sequence to string representation
    static std::string toString(const Sequence& sequence);
    /// Convert Base enum to character
    static inline char toChar(Base base);

    /// Get raw packed data buffer (mutable)
    inline Word* buffer() noexcept { return data; }
    /// Get raw packed data buffer (const)
    inline const Word* buffer() const noexcept { return data; }
    /// Number of storage units in the buffer
    inline size_t unitCount() const noexcept { return (length() + BasesPerWord - 1) >> WordShift; }
private:
    //--- Private data and helpers ---
    Sequence *parent;  ///< Optional parent sequence for overlays
    Word *data;   ///< Packed base data storage

    /// Calculate index of block containing position
    static inline size_t index(size_t pos) noexcept {
        return pos >> WordShift;
    }
    /// Calculate bit shift within block for position
    static inline size_t shift(size_t pos) noexcept {
        return (pos & WordMask) * BaseLengthBits;
    }

    /// Lookup table to map ASCII characters to Base enum
    static inline constexpr Base baseTable[256] = {
            N, N, N, N, N, N, N, N, // 0-7
            N, N, N, N, N, N, N, N, // 8-15
            N, N, N, N, N, N, N, N, // 16-23
            N, N, N, N, N, N, N, N, // 24-31
            N, N, N, N, N, N, N, N, // 32-39
            N, N, N, N, N, N, N, N, // 40-47
            N, N, N, N, N, N, N, N, // 48-55
            N, N, N, N, N, N, N, N, // 56-63
            N, A, N, C, N, N, G, N, // 64-71 ('A'=65, 'C'=67, 'G'=71)
            N, N, N, N, N, N, N, N, // 72-79
            N, N, N, N, N, N, N, T, // 80-87 ('T'=84)
            N, N, N, N, N, N, N, N, // 88-95
            N, N, N, N, N, N, N, N, // 96-103
            A, N, C, N, N, N, G, N, // 104-111 ('a'=97, 'c'=99, 'g'=103)
            N, N, N, N, N, N, N, T, // 112-119 ('t'=116)
            N, N, N, N, N, N, N, N, // 120-127
            N, N, N, N, N, N, N, N, // 128-135
            N, N, N, N, N, N, N, N, // 136-143
            N, N, N, N, N, N, N, N, // 144-151
            N, N, N, N, N, N, N, N, // 152-159
            N, N, N, N, N, N, N, N, // 160-167
            N, N, N, N, N, N, N, N, // 168-175
            N, N, N, N, N, N, N, N, // 176-183
            N, N, N, N, N, N, N, N, // 184-191
            N, N, N, N, N, N, N, N, // 192-199
            N, N, N, N, N, N, N, N, // 200-207
            N, N, N, N, N, N, N, N, // 208-215
            N, N, N, N, N, N, N, N, // 216-223
            N, N, N, N, N, N, N, N, // 224-231
            N, N, N, N, N, N, N, N, // 232-239
            N, N, N, N, N, N, N, N, // 240-247
            N, N, N, N, N, N, N, N, // 248-255
    };
};

template<typename Word>
Sequence<Word>::Sequence(Feature feature, Sequence *parent)
        : Feature(feature), parent(parent) {
    this->data = nullptr;
}

template<typename Word>
Sequence<Word>::Sequence(const std::string &sequenceString)
        : Feature(0, sequenceString.length(), Strand::UNDEFINED), parent(nullptr) {
    // allocate enough blocks to hold 'end' bases
    size_t nunits = (end + BasesPerWord - 1) >> WordShift;
    data = static_cast<Word*>(std::aligned_alloc(
        alignof(Word), nunits * sizeof(Word)
    ));
    if (!data) throw std::bad_alloc();
    // clear the final partial block only (others will be overwritten)
    if (end & WordMask) data[nunits - 1] = 0;
    for (size_t n = 0; n < end; ++n) {
        size_t i = index(n), s = shift(n);
        // pack each base into the block array
        this->data[i] |= (Word)baseTable[static_cast<unsigned char>(sequenceString[n])] << s;
    }
}

// Get Base Operator
template<typename Word>
typename Sequence<Word>::Base Sequence<Word>::operator()(size_t pos) const {
    pos += start;
    if (pos >= end) { throw std::out_of_range("Position out of range"); }
    Sequence *current = this;
    while (current->parent) current = current->parent;
    return (Base) ((current->data[index(pos)] >> shift(pos)) & BaseMask);
}

// Set Base Operator
template<typename Word>
void Sequence<Word>::operator()(size_t pos, Base base) {
    pos += start;
    assert(pos < end);
    size_t i = index(pos), s = shift(pos);
    Sequence *current = this;
    while (current->parent) current = current->parent;
    current->data[i] &= ~(BaseMask << s);
    current->data[i] |= ((Word) base) << s;
}

// Subsequence Get Operator
template<typename Word>
Sequence<Word> Sequence<Word>::operator[](Feature feature) { return Sequence(feature , this); }

// Subsequence Set Operator
template<typename Word>
void Sequence<Word>::operator[](Feature feature, Sequence sequence) {
    if (feature.length() != sequence.length()) {
        throw std::invalid_argument("Feature and sequence lengths do not match");
    }
    for (size_t i = 0; i < feature.length(); ++i) {
        (*this)(feature.start + i, sequence(i));
    }
}

template<typename Word>
char Sequence<Word>::toChar(Base base) {
    switch(base) {
        case A: return 'A';
        case C: return 'C';
        case G: return 'G';
        case T: return 'T';
            [[unlikely]] default:
            return 'N';
    }
}

template<typename Word>
std::string Sequence<Word>::toString(const Sequence& sequence) {
    std::string sequenceString;
    size_t length = sequence.length();
    sequenceString.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        sequenceString.push_back(toChar(sequence[i]));
    }
    return sequenceString;
}

//--- End of class Sequence ---
} // namespace glasshelix

#endif //GLASSHELIX_SEQUENCE_HH
