#ifndef TIMER_CPU_HPP__
#define TIMER_CPU_HPP__



#include <chrono>
#include <string>


class CpuTimer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> clock_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> clock_stop;
public:
    CpuTimer() {}
    void start();
    void stop();
    float elapsed();
    void printElapsed(
        std::string timed_fct_name
    );
};



#endif
