#ifndef HISTOGRAM_HPP__
#define HISTOGRAM_HPP__



// cuda file
void thrustHistogramWithBinIdxs(
    const float* const d_in,
    const unsigned int length,
    const unsigned int n_bins,
    unsigned int* const d_bins,
    unsigned int* const d_bin_idxs
);

void parallelHistogramWithBinIdxs(
    const float* const d_in,
    const unsigned int length,
    const unsigned int n_bins,
    unsigned int* const d_bins,
    unsigned int* const d_bin_idxs
);


// cpp file
void serialHistogramWithBinIdxs(
    const float* const h_in,
    const unsigned int length,
    const unsigned int n_bins,
    unsigned int* const h_bins,
    unsigned int* const h_bin_idxs
);



#endif
