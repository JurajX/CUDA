#ifndef CUDA_MEMORY_HPP__
#define CUDA_MEMORY_HPP__



void allocCudaMem(
    void** d_DataPtrPtr,
    const unsigned int size
);

void gpuMemFree(
    void** d_DataPtrPtr
);

void memsetZero(
    void* d_DataPtr,
    const unsigned int size
);

void memcpyCPUtoGPU(
    void* h_DataPtr,
    void* d_DataPtr,
    const unsigned int size
);

void memcpyGPUtoCPU(
    void* d_DataPtr,
    void* h_DataPtr,
    const unsigned int size
);

void memcpyGPUtoGPU(
    void* d_DataFromPtr,
    void* d_DataToPtr,
    const unsigned int size
);



#endif
