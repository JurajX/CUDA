#include "cudaMemory.hpp"

#include <cuda.h>
#include "checkCudaErrors.hpp"


void allocCudaMem(
    void** d_DataPtrPtr,
    const unsigned int size
) {
    checkCudaErrors(cudaMalloc(d_DataPtrPtr, size));
    checkCudaErrors(cudaMemset(*d_DataPtrPtr, 0, size));
}

void gpuMemFree(
    void** d_DataPtrPtr
) {
    checkCudaErrors(cudaFree(*d_DataPtrPtr));
    *d_DataPtrPtr = nullptr;
}

void memsetZero(
    void* d_DataPtr,
    const unsigned int size
) {
    checkCudaErrors(cudaMemset(d_DataPtr, 0, size));
}

void memcpyCPUtoGPU(
    void* h_DataPtr,
    void* d_DataPtr,
    const unsigned int size
) {
    checkCudaErrors(cudaMemcpy(d_DataPtr, h_DataPtr, size, cudaMemcpyHostToDevice));
}

void memcpyGPUtoCPU(
    void* d_DataPtr,
    void* h_DataPtr,
    const unsigned int size
) {
    checkCudaErrors(cudaMemcpy(h_DataPtr, d_DataPtr, size, cudaMemcpyDeviceToHost));
}

void memcpyGPUtoGPU(
    void* d_DataFromPtr,
    void* d_DataToPtr,
    const unsigned int size
) {
    checkCudaErrors(cudaMemcpy(d_DataToPtr, d_DataFromPtr, size, cudaMemcpyDeviceToDevice));
}
