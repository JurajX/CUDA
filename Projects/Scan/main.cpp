#include <iostream>
#include <memory>
#include <algorithm>

#include "cudaMemory.hpp"
#include "functions.hpp"
#include "timerCPU.hpp"
#include "timerGPU.hpp"

#include "scan.hpp"


void initialise(
    unsigned int* input,
    unsigned int* output,
    unsigned int length
) {
    unsigned int acc = 0;
    output[0] = 0;
    for (unsigned int i = 0; i < length; i++) {
        input[i] = i;
        acc += i;
        output[i+1] += acc;
    }
}


int main(int argc, char const *argv[]) {
    std::cout << "Hello, World! I'm " << argv[0] << "." << std::endl;

    CpuTimer ctimer;
    GpuTimer gtimer;

    // ========== initialisation of arrays
    unsigned int length = 1<<27;
    std::cout << "Initialising arrays of length " << length << "\n";

    auto input  = std::make_unique<unsigned int[]>(length);
    auto output = std::make_unique<unsigned int[]>(length+1);
    bool correct;
    initialise(input.get(), output.get(), length);

    // ========== Scanning arrays
    // allocate memory for working arrays and copy input array to them
    auto h_in = std::make_unique<unsigned int[]>(length);
    std::uninitialized_copy(input.get(), input.get() + length, h_in.get());
    unsigned int *d_in;
    allocCudaMem((void**) &d_in, length*sizeof(unsigned int));                  // gpuMemFree((void**) &d_in);
    memcpyCPUtoGPU((void*) input.get(), (void*) d_in, length*sizeof(unsigned int));

    std::cout << "Scanning arrays\n";

    std::copy(input.get(), input.get() + length, h_in.get());
    ctimer.start();
    serialCPUscan(h_in.get(), length, h_in.get());
    ctimer.stop();
    ctimer.printElapsed("  serial CPU scan");
    correct = checkCorrectness(output.get()+1, h_in.get(), length);
    printCorrectness(correct);


    std::copy(input.get(), input.get() + length, h_in.get());
    ctimer.start();
    stdParallelCPUscan(h_in.get(), length, h_in.get());
    ctimer.stop();
    ctimer.printElapsed("     std CPU scan");
    correct = checkCorrectness(output.get()+1, h_in.get(), length);
    printCorrectness(correct);


    memcpyCPUtoGPU((void*) input.get(), (void*) d_in, length*sizeof(unsigned int));
    gtimer.start();
    parallelGPUscan(d_in, length, d_in);
    gtimer.stop();
    gtimer.printElapsed("parallel GPU scan");
    memcpyGPUtoCPU((void*) d_in, (void*) h_in.get(), length*sizeof(unsigned int));
    correct = checkCorrectness(output.get()+1, h_in.get(), length);
    printCorrectness(correct);


    memcpyCPUtoGPU((void*) input.get(), (void*) d_in, length*sizeof(unsigned int));
    gtimer.start();
    thrustGPUscan(d_in, length, d_in);
    gtimer.stop();
    gtimer.printElapsed("  thrust GPU scan");
    memcpyGPUtoCPU((void*) d_in, (void*) h_in.get(), length*sizeof(unsigned int));
    correct = checkCorrectness(output.get()+1, h_in.get(), length);
    printCorrectness(correct);


    gpuMemFree((void**) &d_in);

    std::cout << "Program has finished executing\n";
    return 0;
}
