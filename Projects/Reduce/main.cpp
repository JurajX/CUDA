#include <iostream>
#include <memory>
#include <algorithm>

#include "cudaMemory.hpp"
#include "functions.hpp"
#include "timerCPU.hpp"
#include "timerGPU.hpp"

#include "reduce.hpp"


unsigned int initialise(
    unsigned int* input,
    unsigned int length
) {
    unsigned int sum = 0;
    for (unsigned int i = 0; i < length; i++) {
        input[i] = i;
        sum += i;
    }
    return sum;
}


int main(int argc, char const *argv[]) {
    std::cout << "Hello, World! I'm " << argv[0] << "." << std::endl;

    CpuTimer ctimer;
    GpuTimer gtimer;

    // ========== initialisation of arrays
    unsigned int length = 1<<27;
    std::cout << "Initialising arrays of length " << length << "\n";

    auto input = std::make_unique<unsigned int[]>(length);
    unsigned int sum = initialise(input.get(), length);

    // ========== Reducing arrays
    // allocate memory for working arrays and copy input array to them
    auto h_in = std::make_unique<unsigned int[]>(length);
    std::uninitialized_copy(input.get(), input.get() + length, h_in.get());

    unsigned int *d_in;
    allocCudaMem((void**) &d_in, length*sizeof(unsigned int));                  // gpuMemFree((void**) &d_in);
    memcpyCPUtoGPU((void*) input.get(), (void*) d_in, length*sizeof(unsigned int));

    std::cout << "Reducing arrays\n";

    ctimer.start();
    unsigned int serialCPUresult = serialCPUreduce(h_in.get(), length);
    ctimer.stop();
    ctimer.printElapsed("  serial CPU reduce");
    printCorrectness(sum == serialCPUresult);


    ctimer.start();
    unsigned int stdCPUresult = stdParallelCPUreduce(h_in.get(), length);
    ctimer.stop();
    ctimer.printElapsed("     std CPU reduce");
    printCorrectness(sum == stdCPUresult);


    gtimer.start();
    unsigned int parallelGPUresult = parallelGPUreduce(d_in, length);
    gtimer.stop();
    gtimer.printElapsed("parallel GPU reduce");
    printCorrectness(sum == parallelGPUresult);


    gtimer.start();
    unsigned int thrustGPUresult = thrustGPUreduce(d_in, length);
    gtimer.stop();
    gtimer.printElapsed("  thrust GPU reduce");
    printCorrectness(sum == thrustGPUresult);


    gpuMemFree((void**) &d_in);

    std::cout << "Program has finished executing\n";
    return 0;
}
