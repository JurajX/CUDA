#include "functions.hpp"

#include <iostream>
#include <cassert>
#include <utility>
#include <cmath>


template <typename T>
inline T ceilDiv(
    const T x,
    const T y
) {
    assert(y != 0);
    return x == 0 ? 0 : (x - 1)/(y) + 1;             // ceiling division ceil(x/y) = ((x - 1) / y) + 1
}

unsigned int ui_ceilDiv(
    const unsigned int x,
    const unsigned int y
) {
    return ceilDiv(x, y);
}

unsigned int double_ceilDiv(
    const double x,
    const double y
) {
    assert(y != 0);
    return std::ceil(x/y);
}

void printCorrectness(
    bool correct
) {
    std::cout << "The result was ";
    if (correct)
        std::cout << "correct!\n";
    else
        std::cout << "INCORRECT!\n";
}

template <typename T>
inline bool checkCorrectnessTmplt(
    const T* const first,
    const T* const second,
    const unsigned int length
) {
    for (unsigned int i = 0; i < length; i++) {
        if ( first[i] != second[i] )
            return false;
    }
    return true;
}

bool checkCorrectness(
    const unsigned int* const first,
    const unsigned int* const second,
    const unsigned int length
) {
    return checkCorrectnessTmplt(first, second, length);
}

bool checkCorrectness(
    const std::pair<unsigned int, unsigned int>* const first,
    const std::pair<unsigned int, unsigned int>* const second,
    const unsigned int length
) {
    return checkCorrectnessTmplt(first, second, length);
}
