#include <iostream>
#include <memory>
#include <algorithm>

#include "cudaMemory.hpp"
#include "functions.hpp"
#include "timerCPU.hpp"
#include "timerGPU.hpp"

#include "histogram.hpp"


void initialise(
    float* input,
    unsigned int length
) {
    for (unsigned int i = 0; i < length; i++) {
        input[i] = 1.0F*i;
    }
}


int main(int argc, char const *argv[]) {
    std::cout << "Hello, World! I'm " << argv[0] << "." << std::endl;

    CpuTimer ctimer;
    GpuTimer gtimer;

    // ========== initialisation of arrays
    unsigned int length = 1<<27;
    unsigned int n_bins = 1<<10;
    std::cout << "Initialising arrays of length " << length << "\n";

    auto input    = std::make_unique<float[]>(length);
    auto bin_idxs = std::make_unique<unsigned int[]>(length);
    auto bins     = std::make_unique<unsigned int[]>(n_bins);
    bool correct1, correct2;
    initialise(input.get(), length);

    // get reference array
    ctimer.start();
    serialHistogramWithBinIdxs(input.get(), length, n_bins, bins.get(), bin_idxs.get());
    ctimer.stop();
    ctimer.printElapsed("     std CPU histogram");
    std::cout << '\n';


    // ========== Making Histogram
    // allocate memory for working arrays
    auto h_bin_idxs = std::make_unique<unsigned int[]>(length);
    auto h_bins    = std::make_unique<unsigned int[]>(n_bins);

    float *d_in;
    allocCudaMem((void**) &d_in, length*sizeof(float));                 // gpuMemFree((void**) &d_in);
    memcpyCPUtoGPU((void*) input.get(), (void*) d_in, length*sizeof(float));
    unsigned int *d_bin_idxs;
    allocCudaMem((void**) &d_bin_idxs, length*sizeof(unsigned int));    // gpuMemFree((void**) &d_bin_idxs);
    unsigned int *d_bins;
    allocCudaMem((void**) &d_bins, n_bins*sizeof(unsigned int));        // gpuMemFree((void**) &d_bins);


    std::cout << "Making Histogram\n";

    gtimer.start();
    parallelHistogramWithBinIdxs(d_in, length, n_bins, d_bins, d_bin_idxs);
    gtimer.stop();
    gtimer.printElapsed("parallel GPU histogram");
    memcpyGPUtoCPU((void*) d_bin_idxs, (void*) h_bin_idxs.get(), length*sizeof(unsigned int));
    memcpyGPUtoCPU((void*) d_bins,     (void*) h_bins.get(),     n_bins*sizeof(unsigned int));
    correct1 = checkCorrectness(bin_idxs.get(), h_bin_idxs.get(), length);
    correct2 = checkCorrectness(bins.get(),     h_bins.get(),     n_bins);
    printCorrectness(correct1 && correct2);


    memsetZero((void*) d_bin_idxs, length*sizeof(unsigned int));
    memsetZero((void*) d_bins,     n_bins*sizeof(unsigned int));
    gtimer.start();
    thrustHistogramWithBinIdxs(d_in, length, n_bins, d_bins, d_bin_idxs);
    gtimer.stop();
    gtimer.printElapsed("  thrust GPU histogram");
    memcpyGPUtoCPU((void*) d_bin_idxs, (void*) h_bin_idxs.get(), length*sizeof(unsigned int));
    memcpyGPUtoCPU((void*) d_bins,     (void*) h_bins.get(),     n_bins*sizeof(unsigned int));
    correct1 = checkCorrectness(bin_idxs.get(), h_bin_idxs.get(), length);
    correct2 = checkCorrectness(bins.get(),     h_bins.get(),     n_bins);
    printCorrectness(correct1 && correct2);

    gpuMemFree((void**) &d_in);
    gpuMemFree((void**) &d_bin_idxs);
    gpuMemFree((void**) &d_bins);

    std::cout << "Program has finished executing\n";
    return 0;
}
