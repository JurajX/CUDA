#include "reduce.hpp"

#include <numeric>
#include <execution>
#include <algorithm>
#include <limits>

#include "functions.hpp"


// -------------------- CPU Serial Reduce --------------------
template <typename T>
T sCPUreduce(
    const T* const h_in,
    const unsigned int length,
    const T identity,
    T (*fct)(const T, const T)
) {
    T ret = identity;
    for (unsigned int i = 0; i < length; i++) {
        ret = fct(ret, h_in[i]);
    }
    return ret;
}


unsigned int serialCPUreduce(
    const unsigned int* const h_in,
    const unsigned int length
) {
    return sCPUreduce(h_in, length, 0U, addition);
}


// -------------------- CPU Parallel Reduce --------------------
unsigned int stdParallelCPUreduce(
    const unsigned int* const h_in,
    const unsigned int length
) {
    return std::reduce(std::execution::par, h_in, h_in + length, 0U, std::plus<>());
}


// -------------------- CPU Serial Min Max --------------------
template <typename T>
inline void sCPUfindMinMax(
    const T* const h_in,
    const unsigned int length,
    T& min,
    T& max
) {
    for (unsigned int i = 0; i < length; i++) {
        min = std::min(min, h_in[i]);
        max = std::max(max, h_in[i]);
    }
}

void serialCPUfindMinMaxFloat(
    const float* const h_in,
    const unsigned int length,
    float& min,
    float& max
) {
    min =  std::numeric_limits<float>::max();
    max = -std::numeric_limits<float>::max();
    sCPUfindMinMax(h_in, length, min, max);
}
