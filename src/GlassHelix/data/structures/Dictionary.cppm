export module Dictionary;

import BidirectionalMap;

import <string>;

export
template<typename T>
class Dictionary : virtual public IDMap<T> {
public:
    explicit Dictionary(std::string filename = "") {

    }
};
