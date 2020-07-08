#include "reduce.hpp"

#include "checkCudaErrors.hpp"
#include "cudaMemory.hpp"
#include "functions.hpp"

#include <cuda.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

#include <stdexcept>
#include <string>
#include <limits>
#include <utility>


// -------------------- GPU Parallel Reduce Add (thrust) --------------------
unsigned int thrustGPUreduce(
    const unsigned int* const d_in,
    const unsigned int length
) {
    return thrust::reduce(thrust::device, d_in, d_in + length, 0, thrust::plus<unsigned int>());
}


// -------------------- GPU Parallel Find Min Max (thrust) --------------------
void thrustGPUfindMinMaxFloat(
    const float* const d_in,
    const unsigned int length,
    float& min,
    float& max
) {
    thrust::pair<const float*, const float*> result = thrust::minmax_element(thrust::device, d_in, d_in + length);
    memcpyGPUtoCPU((void*) result.first,  (void*) &min, sizeof(float));
    memcpyGPUtoCPU((void*) result.second, (void*) &max, sizeof(float));
}


// -------------------- GPU Parallel Reduce Add --------------------
template <typename T>
__device__ __forceinline__ void warpAdd(
    volatile T* sh_data,
    const unsigned int idx
) {
    sh_data[idx] = sh_data[idx] + sh_data[idx + 32];
    sh_data[idx] = sh_data[idx] + sh_data[idx + 16];
    sh_data[idx] = sh_data[idx] + sh_data[idx +  8];
    sh_data[idx] = sh_data[idx] + sh_data[idx +  4];
    sh_data[idx] = sh_data[idx] + sh_data[idx +  2];
    sh_data[idx] = sh_data[idx] + sh_data[idx +  1];
}


template <typename T>
__global__ void kernelReduceAdd(
    const T* const d_in,
    const unsigned int length,
    const T identity,
    T* const d_out
) {
    const unsigned int absIdx = blockIdx.x*blockDim.x*4 + threadIdx.x;
    const unsigned int idx = threadIdx.x;

    extern __shared__ __align__(sizeof(T)) unsigned char sh_mem[];
    T* sh_data = reinterpret_cast<T*>(sh_mem);

    if (absIdx < length) {
        sh_data[idx] = d_in[absIdx];
    } else {
        sh_data[idx] = identity;
    }
    for (unsigned int i = 1; i < 4; i++) {
        if (absIdx + i*blockDim.x < length)
            sh_data[idx] = sh_data[idx] + d_in[ absIdx + i*blockDim.x ];
    }
    __syncthreads();

    for (unsigned int i = blockDim.x/4; i > 32; i >>= 2) {
        if (idx >= i)
            return;
        sh_data[idx] = sh_data[idx] + sh_data[ 3*idx + i ] + sh_data[ 3*idx + i + 1 ] + sh_data[ 3*idx + i + 2 ];
        __syncthreads();
    }
    if (idx >= warpSize)
        return;
    warpAdd(sh_data, idx);
    if (idx == 0)
        d_out[blockIdx.x] = sh_data[0];
}


template <typename T>
T pGPUreduce(
    const T* const d_in,
    const unsigned int length,
    const T identity
) {
    dim3 blockDim(1024, 1, 1);
    unsigned int gridX = ui_ceilDiv(length, 4*blockDim.x);
    dim3 gridDim(gridX, 1, 1);

    T* d_o;
    T* d_i;
    allocCudaMem((void**) &d_o,            gridDim.x               *sizeof(T));     // gpuMemFree((void**) &d_o);
    allocCudaMem((void**) &d_i, ui_ceilDiv(gridDim.x, 4*blockDim.x)*sizeof(T));     // gpuMemFree((void**) &d_i);

    kernelReduceAdd<<<gridDim, blockDim, blockDim.x*sizeof(T)>>>(d_in, length, identity, d_o);
    while (gridDim.x > 1) {
        std::swap(d_o, d_i);
        kernelReduceAdd<<<gridDim, blockDim, blockDim.x*sizeof(T)>>>(d_i, gridDim.x, identity, d_o);
        gridDim.x = ui_ceilDiv(gridDim.x, 4*blockDim.x);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    T ret;
    memcpyGPUtoCPU((void*) d_o, (void*) &ret, sizeof(T));

    gpuMemFree((void**) &d_i);
    gpuMemFree((void**) &d_o);

    return ret;
}


unsigned int parallelGPUreduce(
    const unsigned int* const d_in,
    const unsigned int length
) {
    return pGPUreduce(d_in, length, 0U);
}


// -------------------- GPU Parallel Find Min Max --------------------
template <typename T>
__device__ __forceinline__ void warpMin(
    volatile T* sh_data,
    const unsigned int idx
) {
    sh_data[idx] = min( sh_data[idx], sh_data[idx + 32] );
    sh_data[idx] = min( sh_data[idx], sh_data[idx + 16] );
    sh_data[idx] = min( sh_data[idx], sh_data[idx +  8] );
    sh_data[idx] = min( sh_data[idx], sh_data[idx +  4] );
    sh_data[idx] = min( sh_data[idx], sh_data[idx +  2] );
    sh_data[idx] = min( sh_data[idx], sh_data[idx +  1] );
}


template <typename T>
__global__ void kernelReduceMin(
    const T* const d_in,
    const unsigned int length,
    const T identity,
    T* const d_out
) {
    const unsigned int absIdx = blockIdx.x*blockDim.x*4 + threadIdx.x;
    const unsigned int idx = threadIdx.x;

    extern __shared__ __align__(sizeof(T)) unsigned char sh_mem[];
    T* sh_data = reinterpret_cast<T*>(sh_mem);

    if (absIdx < length) {
        sh_data[idx] = d_in[absIdx];
    } else {
        sh_data[idx] = identity;
    }
    for (unsigned int i = 1; i < 4; i++) {
        if (absIdx + i*blockDim.x < length)
            sh_data[idx] = min(sh_data[idx], d_in[ absIdx + i*blockDim.x ]);
    }
    __syncthreads();

    for (unsigned int i = blockDim.x/4; i > 32; i >>= 2) {
        if (idx >= i)
            return;
        sh_data[idx] = min( min( sh_data[idx], sh_data[ 3*idx + i ] ), min( sh_data[ 3*idx + i + 1 ], sh_data[ 3*idx + i + 2 ] ) );
        __syncthreads();
    }
    if (idx >= warpSize)
        return;
    warpMin(sh_data, idx);
    if (idx == 0)
        d_out[blockIdx.x] = sh_data[0];
}


template <typename T>
__device__ __forceinline__ void warpMax(
    volatile T* sh_data,
    const unsigned int idx
) {
    sh_data[idx] = max( sh_data[idx], sh_data[idx + 32] );
    sh_data[idx] = max( sh_data[idx], sh_data[idx + 16] );
    sh_data[idx] = max( sh_data[idx], sh_data[idx +  8] );
    sh_data[idx] = max( sh_data[idx], sh_data[idx +  4] );
    sh_data[idx] = max( sh_data[idx], sh_data[idx +  2] );
    sh_data[idx] = max( sh_data[idx], sh_data[idx +  1] );
}


template <typename T>
__global__ void kernelReduceMax(
    const T* const d_in,
    const unsigned int length,
    const T identity,
    T* const d_out
) {
    const unsigned int absIdx = blockIdx.x*blockDim.x*4 + threadIdx.x;
    const unsigned int idx = threadIdx.x;

    extern __shared__ __align__(sizeof(T)) unsigned char sh_mem[];
    T* sh_data = reinterpret_cast<T*>(sh_mem);

    if (absIdx < length) {
        sh_data[idx] = d_in[absIdx];
    } else {
        sh_data[idx] = identity;
    }
    for (unsigned int i = 1; i < 4; i++) {
        if (absIdx + i*blockDim.x < length)
            sh_data[idx] = max(sh_data[idx], d_in[ absIdx + i*blockDim.x ]);
    }
    __syncthreads();

    for (unsigned int i = blockDim.x/4; i > 32; i >>= 2) {
        if (idx >= i)
            return;
        sh_data[idx] = max( max( sh_data[idx], sh_data[ 3*idx + i ] ), max( sh_data[ 3*idx + i + 1 ], sh_data[ 3*idx + i + 2 ] ) );
        __syncthreads();
    }
    if (idx >= warpSize)
        return;
    warpMax(sh_data, idx);
    if (idx == 0)
        d_out[blockIdx.x] = sh_data[0];
}


template <typename T>
T pGPUfindMinMax(
    const T* const d_in,
    const unsigned int length,
    const T identity,
    std::string op
) {
    void (*kernel)(const T* const, const unsigned int, const T, T* const) = nullptr;
    if (op == "min") {
        kernel = kernelReduceMin<T>;
    } else if (op == "max") {
        kernel = kernelReduceMax<T>;
    } else {
        throw std::invalid_argument("Operators supported by parallelReduce are: min or max. Given: " + op);
    }

    dim3 blockDim(1024, 1, 1);
    unsigned int gridX = ui_ceilDiv(length, 4*blockDim.x);
    dim3 gridDim(gridX, 1, 1);

    T* d_o;
    T* d_i;
    allocCudaMem((void**) &d_o,            gridDim.x               *sizeof(T));     // gpuMemFree((void**) &d_o);
    allocCudaMem((void**) &d_i, ui_ceilDiv(gridDim.x, 4*blockDim.x)*sizeof(T));     // gpuMemFree((void**) &d_i);

    kernel<<<gridDim, blockDim, blockDim.x*sizeof(T)>>>(d_in, length, identity, d_o);
    while (gridDim.x > 1) {
        std::swap(d_o, d_i);
        kernel<<<gridDim, blockDim, blockDim.x*sizeof(T)>>>(d_i, gridDim.x, identity, d_o);
        gridDim.x = ui_ceilDiv(gridDim.x, 4*blockDim.x);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    T ret;
    memcpyGPUtoCPU((void*) d_o, (void*) &ret, sizeof(T));

    gpuMemFree((void**) &d_i);
    gpuMemFree((void**) &d_o);

    return ret;
}


void parallelGPUfindMinMaxFloat(
    const float* const d_in,
    const float length,
    float& min,
    float& max
) {
    const float flt_max = std::numeric_limits<float>::max();
    min = pGPUfindMinMax(d_in, length,  flt_max, "min");
    max = pGPUfindMinMax(d_in, length, -flt_max, "max");
}


// -------------------- GPU Parallel Reduce Bit Or --------------------
template <typename T>
__device__ __forceinline__ void warpBitOr(
    volatile T* sh_data,
    const unsigned int idx
) {
    sh_data[idx] = sh_data[idx] | sh_data[idx + 32];
    sh_data[idx] = sh_data[idx] | sh_data[idx + 16];
    sh_data[idx] = sh_data[idx] | sh_data[idx +  8];
    sh_data[idx] = sh_data[idx] | sh_data[idx +  4];
    sh_data[idx] = sh_data[idx] | sh_data[idx +  2];
    sh_data[idx] = sh_data[idx] | sh_data[idx +  1];
}

__device__ __forceinline__ unsigned int getKey(
    unsigned int key
) {
    return key;
}
__device__ __forceinline__ unsigned int getKey(
    thrust::pair<unsigned int, unsigned int> element
) {
    return element.first;
}

__device__ __forceinline__ unsigned int storeKey(
    unsigned int key,
    unsigned int* d_out
) {
    return key;
}
__device__ __forceinline__ thrust::pair<unsigned int, unsigned int> storeKey(
    unsigned int key,
    thrust::pair<unsigned int, unsigned int>* d_out
) {
    return thrust::pair<unsigned int, unsigned int>(key, 0);
}

template <typename T>
__global__ void kernelReduceBitOr(
    const T* const d_in,
    const unsigned int length,
    const T identity,
    T* const d_out
) {
    const unsigned int absIdx = blockIdx.x*blockDim.x*4 + threadIdx.x;
    const unsigned int idx = threadIdx.x;

    extern __shared__ __align__(sizeof(unsigned int)) unsigned char sh_mem[];
    unsigned int* sh_data = reinterpret_cast<unsigned int*>(sh_mem);

    T element;
    if (absIdx < length) {
        element = d_in[absIdx];
    } else {
        element = identity;
    }
    sh_data[idx] = getKey(element);

    for (unsigned int i = 1; i < 4; i++) {
        if (absIdx + i*blockDim.x < length)
            sh_data[idx] = sh_data[idx] | getKey(d_in[ absIdx + i*blockDim.x ]);
    }
    __syncthreads();

    for (unsigned int i = blockDim.x/4; i > 32; i >>= 2) {
        if (idx >= i)
            return;
        sh_data[idx] = sh_data[idx] | sh_data[ 3*idx + i ] | sh_data[ 3*idx + i + 1 ] | sh_data[ 3*idx + i + 2 ];
        __syncthreads();
    }
    if (idx >= warpSize)
        return;
    warpBitOr(sh_data, idx);
    if (idx == 0)
        d_out[blockIdx.x] = storeKey(sh_data[0], d_out);
}


template <typename T>
T pGPUreduceBitOr(
    const T* const d_in,
    const unsigned int length,
    const T identity
) {
    dim3 blockDim(1024, 1, 1);
    unsigned int gridX = ui_ceilDiv(length, 4*blockDim.x);
    dim3 gridDim(gridX, 1, 1);

    T* d_o;
    T* d_i;
    allocCudaMem((void**) &d_o,            gridDim.x               *sizeof(T));     // gpuMemFree((void**) &d_o);
    allocCudaMem((void**) &d_i, ui_ceilDiv(gridDim.x, 4*blockDim.x)*sizeof(T));     // gpuMemFree((void**) &d_i);

    kernelReduceBitOr<<<gridDim, blockDim, blockDim.x*sizeof(T)>>>(d_in, length, identity, d_o);
    while (gridDim.x > 1) {
        std::swap(d_o, d_i);
        kernelReduceBitOr<<<gridDim, blockDim, blockDim.x*sizeof(T)>>>(d_i, gridDim.x, identity, d_o);
        gridDim.x = ui_ceilDiv(gridDim.x, 4*blockDim.x);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    T ret;
    memcpyGPUtoCPU((void*) d_o, (void*) &ret, sizeof(T));

    gpuMemFree((void**) &d_i);
    gpuMemFree((void**) &d_o);

    return ret;
}


unsigned int parallelGPUreduceBitOr(
    unsigned int* d_in,
    const unsigned int length
) {
    return pGPUreduceBitOr((unsigned int*) d_in, length, 0U);
}


unsigned int parallelGPUreduceBitOr(
    std::pair<unsigned int, unsigned int>* d_in,
    const unsigned int length
) {
    auto identity = thrust::pair<unsigned int, unsigned int>(0U, 0U);
    return pGPUreduceBitOr((thrust::pair<unsigned int, unsigned int>*) d_in, length, identity).first;
}
