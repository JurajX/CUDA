#include "scan.hpp"

#include "checkCudaErrors.hpp"
#include "cudaMemory.hpp"
#include "functions.hpp"

#include <cuda.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>


// -------------------- GPU Parallel Reduce Add (thrust) --------------------
void thrustGPUscan(
    const unsigned int* const d_in,
    const unsigned int length,
    unsigned int* const d_out
) {
    thrust::inclusive_scan(thrust::device, d_in, d_in + length, d_out, thrust::plus<unsigned int>());
}


// -------------------- GPU Parallel Reduce Add --------------------
template <typename T>
__global__ void kernelIncrementEach(
    T* const d_out,
    const T* const d_increment,
    const unsigned int length
) {
    unsigned int absIdx = (blockIdx.x + 1)*blockDim.x + threadIdx.x;    // first 1024 elements don't need to do any more work
    if (absIdx >= length)
        return;
    d_out[absIdx] += d_increment[blockIdx.x];
}


template <typename T>
__global__ void kernelHillisSteelScanAdd(
    T* const d_in,
    const unsigned int length,
    T* const d_out,
    T* const d_increment
) {
    extern __shared__ T sh_data[];                       // double buffer allows for read-modify-write op with only one __syncthreads()

    unsigned int absIdx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idx = threadIdx.x;
    unsigned int bDim = blockDim.x;
    unsigned int buffIn = bDim, buffOut = 0;

    if (idx == 0) {
        sh_data[idx] = d_in[absIdx];
    } else if (absIdx >= length) {
        return;
    } else {
        sh_data[buffOut + idx] = d_in[absIdx - 1] + d_in[absIdx];
    }
    __syncthreads();

    for (unsigned int i = 2; i < blockDim.x; i <<= 1) {
        buffOut = bDim - buffOut;
        buffIn  = bDim - buffIn;

        if (idx < i) {
            sh_data[buffOut + idx] = sh_data[buffIn + idx];
            d_out[absIdx] = sh_data[buffOut + idx];
            return;
        } else {
            sh_data[buffOut + idx] = sh_data[buffIn + idx] + sh_data[buffIn + idx - i];
        }
        __syncthreads();
    }
    d_out[absIdx] = sh_data[buffOut + idx];
    if (idx == blockDim.x - 1)
        d_increment[blockIdx.x] = sh_data[buffOut + idx];
}


template <typename T>
void pGPUscan(
    T* const d_in,
    const unsigned int length,
    T* const d_out,
    const unsigned int block_dim
) {
    dim3 blockDim(block_dim, 1, 1);
    unsigned int n = ui_ceilDiv(length, blockDim.x);
    unsigned int steps = double_ceilDiv( std::log2(n), std::log2(blockDim.x) ) + 1;

    T* d_inters[steps+1];
    T* d_outs[steps+1];
    const unsigned int* gridXs[steps+1];

    d_inters[0] = d_in;
    d_outs[0] = d_out;
    gridXs[0] = &length;

    for (unsigned int i = 1; i <= steps; i++) {
        unsigned int* gridX = new unsigned int;
        *gridX = ui_ceilDiv(*gridXs[i-1], blockDim.x);
        gridXs[i] = gridX;
        dim3 gridDim(*gridXs[i], 1, 1);

        T* d_increment;
        T* d_outNext;                                                   // for the next run, the last will be unused
        allocCudaMem((void**) &d_increment, gridDim.x*sizeof(unsigned int));     // gpuMemFree((void**) &d_increment);
        allocCudaMem((void**) &d_outNext,   gridDim.x*sizeof(unsigned int));     // gpuMemFree((void**) &d_outNext);
        d_inters[i] = d_increment;
        d_outs[i]   = d_outNext;

        kernelHillisSteelScanAdd<<<gridDim, blockDim, blockDim.x*2*sizeof(T)>>>(d_inters[i-1], *gridXs[i-1], d_outs[i-1], d_inters[i]);
    }

    for (unsigned int i = steps-1; i >= 1; i--) {
        dim3 gridDim(*gridXs[i]-1, 1, 1);                               // first 1024 elements don't need an update, so lunch 1 block less
        kernelIncrementEach<<<gridDim, blockDim>>>(d_outs[i-1], d_outs[i], *gridXs[i-1]);
    }

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    for (unsigned int i = 1; i <= steps; i++) {
        delete gridXs[i];
        gpuMemFree((void**) &d_inters[i]);
        gpuMemFree((void**) &d_outs[i]);
    }
}


void parallelGPUscan(
    unsigned int* const d_in,
    const unsigned int length,
    unsigned int* const d_out
) {
    pGPUscan(d_in, length, d_out, 128);
}
