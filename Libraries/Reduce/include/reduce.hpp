#ifndef REDUCE_HPP__
#define REDUCE_HPP__



#include <utility>


// cuda file
unsigned int thrustGPUreduce(
    const unsigned int* const d_in,
    const unsigned int length
);

void thrustGPUfindMinMaxFloat(
    const float* const d_in,
    const unsigned int length,
    float& min,
    float& max
);

unsigned int parallelGPUreduce(
    const unsigned int* const d_in,
    const unsigned int length
);

void parallelGPUfindMinMaxFloat(
    const float* const d_in,
    const float length,
    float& min,
    float& max
);

unsigned int parallelGPUreduceBitOr(
    unsigned int*  d_in,
    const unsigned int length
);

unsigned int parallelGPUreduceBitOr(
    std::pair<unsigned int, unsigned int>*  d_in,
    const unsigned int length
);


// cpp file
unsigned int serialCPUreduce(
    const unsigned int* const h_in,
    const unsigned int length
);

unsigned int stdParallelCPUreduce(
    const unsigned int* const h_in,
    const unsigned int length
);

void serialCPUfindMinMaxFloat(
    const float* const h_in,
    const unsigned int length,
    float& min,
    float& max
);


#endif
