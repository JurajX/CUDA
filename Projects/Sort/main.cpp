#include <iostream>
#include <memory>
#include <algorithm>
#include <utility>

#include "cudaMemory.hpp"
#include "functions.hpp"
#include "timerCPU.hpp"
#include "timerGPU.hpp"

#include "sort.hpp"


void initialise(
    unsigned int* input,
    unsigned int* output,
    std::pair<unsigned int, unsigned int>* kv_input,
    std::pair<unsigned int, unsigned int>* kv_output,
    unsigned int length
) {
    for (unsigned int i = 0; i < length; i++) {
        input[i]  = length - i;
        output[i] = i + 1;
        kv_input[i]  = std::pair<unsigned int, unsigned int>(length - i, i             );
        kv_output[i] = std::pair<unsigned int, unsigned int>(i + 1,      length - i - 1);
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
    auto output = std::make_unique<unsigned int[]>(length);
    auto kv_input  = std::make_unique< std::pair<unsigned int, unsigned int>[] >(length);
    auto kv_output = std::make_unique< std::pair<unsigned int, unsigned int>[] >(length);
    bool correct;
    initialise(input.get(), output.get(), kv_input.get(), kv_output.get(), length);

    // ========== Sorting arrays
    unsigned int UI_SIZE = sizeof(unsigned int);
    // allocate memory for working arrays
    auto h_in  = std::make_unique<unsigned int[]>(length);
    unsigned int *d_in;
    allocCudaMem((void**) &d_in, length*UI_SIZE);          // gpuMemFree((void**) &d_in);
    memcpyCPUtoGPU((void*) input.get(), (void*) d_in, length*UI_SIZE);

    std::cout << "Sorting arrays\n";

    std::uninitialized_copy(input.get(), input.get() + length, h_in.get());
    ctimer.start();
    serialCPUsort(h_in.get(), length);
    ctimer.stop();
    ctimer.printElapsed("  serial CPU       sort");
    correct = checkCorrectness(output.get(), h_in.get(), length);
    printCorrectness(correct);


    std::copy(input.get(), input.get() + length, h_in.get());
    ctimer.start();
    parallelCPUsort(h_in.get(), length);
    ctimer.stop();
    ctimer.printElapsed("     std CPU       sort");
    correct = checkCorrectness(output.get(), h_in.get(), length);
    printCorrectness(correct);


    memcpyCPUtoGPU((void*) input.get(), (void*) d_in, length*UI_SIZE);
    gtimer.start();
    parallelGPUsort(d_in, length);
    gtimer.stop();
    gtimer.printElapsed("parallel GPU merge sort");
    memcpyGPUtoCPU((void*) d_in, (void*) h_in.get(), length*UI_SIZE);
    correct = checkCorrectness(output.get(), h_in.get(), length);
    printCorrectness(correct);


    memcpyCPUtoGPU((void*) input.get(), (void*) d_in, length*UI_SIZE);
    gtimer.start();
    parallelGPUradixsort(d_in, length);
    gtimer.stop();
    gtimer.printElapsed("parallel GPU radix sort");
    memcpyGPUtoCPU((void*) d_in, (void*) h_in.get(), length*UI_SIZE);
    correct = checkCorrectness(output.get(), h_in.get(), length);
    printCorrectness(correct);


    memcpyCPUtoGPU((void*) input.get(), (void*) d_in, length*UI_SIZE);
    gtimer.start();
    thrustGPUsort(d_in, length);
    gtimer.stop();
    gtimer.printElapsed("  thrust GPU       sort");
    memcpyGPUtoCPU((void*) d_in, (void*) h_in.get(), length*UI_SIZE);
    correct = checkCorrectness(output.get(), h_in.get(), length);
    printCorrectness(correct);

    gpuMemFree((void**) &d_in);


    // ========== sorting key-value pair arrays
    unsigned int PAIR_UI_SIZE = sizeof(std::pair<unsigned int, unsigned int>);
    // allocate memory for working arrays
    auto h_inPair  = std::make_unique< std::pair<unsigned int, unsigned int>[] >(length);
    std::pair<unsigned int, unsigned int> *d_inPair;
    allocCudaMem((void**) &d_inPair, length*PAIR_UI_SIZE);          // gpuMemFree((void**) &d_inPair);
    memcpyCPUtoGPU((void*) kv_input.get(), (void*) d_inPair, length*PAIR_UI_SIZE);

    std::cout << "Sorting key-value arrays\n";

    std::uninitialized_copy(kv_input.get(), kv_input.get() + length, h_inPair.get());
    ctimer.start();
    serialCPUsort(h_inPair.get(), length);
    ctimer.stop();
    ctimer.printElapsed("  serial CPU       sort");
    correct = checkCorrectness(kv_output.get(), h_inPair.get(), length);
    printCorrectness(correct);


    std::copy(kv_input.get(), kv_input.get() + length, h_inPair.get());
    ctimer.start();
    parallelCPUsort(h_inPair.get(), length);
    ctimer.stop();
    ctimer.printElapsed("     std CPU       sort");
    correct = checkCorrectness(kv_output.get(), h_inPair.get(), length);
    printCorrectness(correct);


    memcpyCPUtoGPU((void*) kv_input.get(), (void*) d_inPair, length*PAIR_UI_SIZE);
    gtimer.start();
    parallelGPUsort(d_inPair, length);
    gtimer.stop();
    gtimer.printElapsed("parallel GPU merge sort");
    memcpyGPUtoCPU((void*) d_inPair, (void*) h_inPair.get(), length*PAIR_UI_SIZE);
    correct = checkCorrectness(kv_output.get(), h_inPair.get(), length);
    printCorrectness(correct);


    memcpyCPUtoGPU((void*) kv_input.get(), (void*) d_inPair, length*PAIR_UI_SIZE);
    gtimer.start();
    parallelGPUradixsort(d_inPair, length);
    gtimer.stop();
    gtimer.printElapsed("parallel GPU radix sort");
    memcpyGPUtoCPU((void*) d_inPair, (void*) h_inPair.get(), length*PAIR_UI_SIZE);
    correct = checkCorrectness(kv_output.get(), h_inPair.get(), length);
    printCorrectness(correct);


    memcpyCPUtoGPU((void*) kv_input.get(), (void*) d_inPair, length*PAIR_UI_SIZE);
    gtimer.start();
    thrustGPUsort(d_inPair, length);
    gtimer.stop();
    gtimer.printElapsed("  thrust GPU       sort");
    memcpyGPUtoCPU((void*) d_inPair, (void*) h_inPair.get(), length*PAIR_UI_SIZE);
    correct = checkCorrectness(kv_output.get(), h_inPair.get(), length);
    printCorrectness(correct);

    gpuMemFree((void**) &d_inPair);


    std::cout << "Program has finished executing\n";
    return 0;
}
