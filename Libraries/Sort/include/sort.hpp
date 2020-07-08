#ifndef SORT_HPP__
#define SORT_HPP__



#include <utility>


// cpp file
void serialCPUsort(
    unsigned int* const h_in,
    const unsigned int length
);

void serialCPUsort(
    std::pair<unsigned int, unsigned int>* const h_in,
    const unsigned int length
);


void parallelCPUsort(
    unsigned int* const h_in,
    const unsigned int length
);

void parallelCPUsort(
    std::pair<unsigned int, unsigned int>* const h_in,
    const unsigned int length
);


// cuda file
void parallelGPUsort(
    unsigned int* const d_in,
    const unsigned int length
);

void parallelGPUsort(
    std::pair<unsigned int, unsigned int>* const d_in,
    const unsigned int length
);


void parallelGPUradixsort(
    unsigned int* const d_in,
    const unsigned int length
);

void parallelGPUradixsort(
    std::pair<unsigned int, unsigned int>* const d_in,
    const unsigned int length
);


void thrustGPUsort(
    unsigned int* const d_in,
    const unsigned int length
);

void thrustGPUsort(
    std::pair<unsigned int, unsigned int>* const d_in,
    const unsigned int length
);


#endif
