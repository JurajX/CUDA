#include "sort.hpp"


#include <algorithm>
#include <execution>


void serialCPUsort(
    unsigned int* const h_in,
    const unsigned int length
) {
    return std::sort(h_in, h_in + length);
}

void serialCPUsort(
    std::pair<unsigned int, unsigned int>* const h_in,
    const unsigned int length
) {
    return std::sort(h_in, h_in + length);
}


void parallelCPUsort(
    unsigned int* const h_in,
    const unsigned int length
) {
    return std::sort(std::execution::par, h_in, h_in + length);
}

void parallelCPUsort(
    std::pair<unsigned int, unsigned int>* const h_in,
    const unsigned int length
) {
    return std::sort(std::execution::par, h_in, h_in + length);
}
