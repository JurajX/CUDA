#ifndef CHECKCUDAERRORS_HPP__
#define CHECKCUDAERRORS_HPP__



#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>


#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T> void check(
    T err,
    const char* const func,
    const char* const file,
    const int line
) {
    if (err != cudaSuccess) {
        std::cout << "CUDA error at: " << file << ":" << line << std::endl;
        std::cout << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}



#endif
