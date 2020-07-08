#ifndef FUNCTIONS_HPP__
#define FUNCTIONS_HPP__



#include <utility>


template <typename T>
inline T addition(
    const T x,
    const T y
) {
    return x + y;
}

unsigned int ui_ceilDiv(
    const unsigned int x,
    const unsigned int y
);

unsigned int double_ceilDiv(
    const double x,
    const double y
);

void printCorrectness(
    bool correct
);

bool checkCorrectness(
    const unsigned int* const first,
    const unsigned int* const second,
    const unsigned int length
);

bool checkCorrectness(
    const std::pair<unsigned int, unsigned int>* const first,
    const std::pair<unsigned int, unsigned int>* const second,
    const unsigned int length
);



#endif
