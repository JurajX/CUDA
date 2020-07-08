#include "timerGPU.hpp"

#include <iostream>
#include <iomanip>
#include <string>

#include <cuda_runtime.h>


GpuTimer::GpuTimer() {
    cudaEventCreate(&clock_start);
    cudaEventCreate(&clock_stop);
}

GpuTimer::~GpuTimer() {
    cudaEventDestroy(clock_start);
    cudaEventDestroy(clock_stop);
}

void GpuTimer::start() {
    cudaEventRecord(clock_start, 0);
}

void GpuTimer::stop() {
    cudaEventRecord(clock_stop, 0);
}

float GpuTimer::elapsed() {
    float elapsed;
    cudaEventSynchronize(clock_stop);
    cudaEventElapsedTime(&elapsed, clock_start, clock_stop);
    return elapsed;
}

void GpuTimer::printElapsed(
    std::string timed_fct_name
) {
    float elapsed = this->elapsed();
    std::cout << "Executing " << timed_fct_name << " took " << std::setw(8) << elapsed << " ms. ";
}
