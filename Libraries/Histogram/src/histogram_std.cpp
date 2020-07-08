#include "histogram.hpp"

#include <algorithm>
#include <limits>

#include "reduce.hpp"


template <typename T>
unsigned int computeBin(
    const T element,
    const unsigned int n_bins,
    const T min,
    const T max
) {
    double binWidth = (1.0/n_bins)*(max - min);
    unsigned int bin = (element - min)/binWidth;
    return (bin >= n_bins)? n_bins - 1 : bin;
}


template <typename T>
void sHistogramWithBinIdxs(
    const T* const h_in,
    const T min,
    const T max,
    const unsigned int length,
    const unsigned int n_bins,
    unsigned int* const h_bins,
    unsigned int* const h_bin_idxs
) {
    for (unsigned int i = 0; i < n_bins; i++) {
        h_bins[i] = 0;
    }

    unsigned int bin;
    for (unsigned int i = 0; i < length; i++) {
        bin = computeBin(h_in[i], n_bins, min, max);
        h_bin_idxs[i] = bin;
        h_bins[bin] += 1;
    }
}


void serialHistogramWithBinIdxs(
    const float* const h_in,
    const unsigned int length,
    const unsigned int n_bins,
    unsigned int* const h_bins,
    unsigned int* const h_bin_idxs
) {
    float min, max;
    serialCPUfindMinMaxFloat(h_in, length, min, max);
    sHistogramWithBinIdxs(h_in, min, max, length, n_bins, h_bins, h_bin_idxs);
}
