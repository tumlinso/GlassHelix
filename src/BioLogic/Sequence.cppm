export module Sequence;

import <cstddef>;
import <string>;

import Feature;

// CHANGE THIS! if you change the size of the blocks or the Base encoding
typedef unsigned long BlockType;
static inline constexpr char BaseLength = 4;
static inline constexpr char BasesPerBlock = sizeof(BlockType) * 8 / BaseLength;

export
typedef enum {
    A = 0b0001,
    C = 0b0010,
    G = 0b0100,
    T = 0b1000,
    N = 0b0000
} Base;

export
class Sequence : public Feature {
    friend class Feature;
public:
    explicit Sequence(Feature feature, Sequence *parent = nullptr);
    explicit Sequence(const std::string &sequenceString);
    ~Sequence() { if (!parent) delete[] data; }

    inline Base operator() (unsigned long pos);
    inline void operator() (unsigned long pos, Base base);

    inline Sequence operator[] (Feature feature);
    inline void operator[] (Feature feature, Sequence sequence);

    static std::string toString(Sequence& sequence);
    static inline char toChar(Base base);
private:
    Sequence *parent;
    BlockType *data;

    static inline unsigned long index(unsigned long pos) { return pos / BasesPerBlock; }
    static inline unsigned long shift(unsigned long pos) { return (pos % BasesPerBlock) * BaseLength; }

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

Sequence::Sequence(Feature feature, Sequence *parent)
        : Feature(feature), parent(parent) {
    this->data = nullptr;
}

Sequence::Sequence(const std::string &sequenceString)
: Feature(0, sequenceString.length(), Feature::Strand::UNDEFINED), parent(nullptr) {
    this->data = new unsigned long[(end * 4 + 63) / 64]();
    for (unsigned long n = 0; n < end; ++n) {
        unsigned long i = index(n), s = shift(n);
        this->data[i] |= ((unsigned long) baseTable[static_cast<unsigned char>(sequenceString[i])]) << s;
    }
}

// Get Base Operator
Base Sequence::operator()(unsigned long pos) {
    pos += start;
    if (pos >= end) { throw std::out_of_range("Position out of range"); }
    Sequence *current = this;
    while (current->parent) current = current->parent;
    return (Base) ((current->data[index(pos)] >> shift(pos)) & 0b1111);
}

// Set Base Operator
void Sequence::operator()(unsigned long pos, Base base) {
    pos += start;
    assert(pos < end);
    unsigned long i = index(pos), s = shift(pos);
    Sequence *current = this;
    while (current->parent) current = current->parent;
    current->data[i] &= ~(0b1111ULL << s);
    current->data[i] |= ((unsigned long) base) << s;
}

// Subsequence Get Operator
Sequence Sequence::operator[](Feature feature) { return Sequence(feature , this); }

// Subsequence Set Operator
void Sequence::operator[](Feature feature, Sequence sequence) {
    if (feature.length() != sequence.length()) {
        throw std::invalid_argument("Feature and sequence lengths do not match");
    }
    for (unsigned long i = 0; i < feature.length(); ++i) {
        (*this)(feature.start + i, sequence(i));
    }
}

char Sequence::toChar(Base base) {
    switch(base) {
        case A: return 'A';
        case C: return 'C';
        case G: return 'G';
        case T: return 'T';
            [[unlikely]] default:
            return 'N';
    }
}

[[maybe_unused]] std::string Sequence::toString(Sequence& sequence) {
    std::string sequenceString;
    unsigned long length = sequence.length();
    sequenceString.reserve(length);
    for (unsigned long i = 0; i < length; ++i) {
        sequenceString.push_back(toChar(sequence[i]));
    }
    return sequenceString;
}
