#ifndef TIMER_GPU_HPP__
#define TIMER_GPU_HPP__



#include <string>

#include <cuda_runtime.h>


class GpuTimer {
private:
    cudaEvent_t clock_start;
    cudaEvent_t clock_stop;
public:
    GpuTimer();
    ~GpuTimer();
    void start();
    void stop();
    float elapsed();
    void printElapsed(
        std::string timed_fct_name
    );
};



#endif
