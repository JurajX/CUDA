#include "timerCPU.hpp"

#include <iostream>
#include <iomanip>
#include <string>


void CpuTimer::start() {
    clock_start = std::chrono::high_resolution_clock::now();
}

void CpuTimer::stop() {
    clock_stop = std::chrono::high_resolution_clock::now();
}

float CpuTimer::elapsed() {
    return ((std::chrono::duration<double>)(clock_stop - clock_start)).count() * 1e3;
}

void CpuTimer::printElapsed(
    std::string timed_fct_name
) {
    float elapsed = this->elapsed();
    std::cout << "Executing " << timed_fct_name << " took " << std::setw(8) << elapsed << " ms. ";
}
