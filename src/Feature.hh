#ifndef GLASSHELIX_FEATURE_HH
#define GLASSHELIX_FEATURE_HH

namespace glasshelix {

    typedef enum struct Strand : signed char {
        SENSE = 1,
        UNDEFINED = 0,
        ANTISENSE = -1
    } Strand;

    class Feature {
    public:
        Strand strand;
        unsigned long start, end;

        Feature(unsigned long start, unsigned long end, Strand strand = Strand::UNDEFINED)
                : start(start), end(end), strand(strand) {};

        [[nodiscard]] inline unsigned long length() const { return end - start; }
    };

}

#endif //GLASSHELIX_FEATURE_HH
