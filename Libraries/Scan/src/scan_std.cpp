#include "scan.hpp"

#include <numeric>
#include <functional>
#include <execution>

#include "functions.hpp"


// -------------------- CPU Serial Scan --------------------
template <typename T>
void serialScanAdd(
    const T* const h_in,
    const unsigned int length,
    const T identity,
    T (*fct)(const T, const T),
    T* const h_out
) {
    T acc = identity;
    for (unsigned int i = 0; i < length; i++) {
        acc = fct(acc, h_in[i]);
        h_out[i] = acc;
    }
}

void serialCPUscan(
    const unsigned int* const h_in,
    const unsigned int length,
    unsigned int* const h_out
) {
    serialScanAdd(h_in, length, 0U, addition, h_out);
}


// -------------------- CPU Parallel Scan --------------------
void stdParallelCPUscan(
    const unsigned int* const h_in,
    const unsigned int length,
    unsigned int* const h_out
) {
    std::inclusive_scan(std::execution::par, h_in, h_in + length, h_out, std::plus<>(), 0U);
}
