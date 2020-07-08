#include "histogram.hpp"

#include "checkCudaErrors.hpp"
#include "cudaMemory.hpp"
#include "functions.hpp"

#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "reduce.hpp"


// -------------------- Make Histogram in Shared Memory --------------------
__global__ void kernelMakeHistogram(
    const unsigned int* const d_bin_idxs,
    unsigned int length,
    unsigned int n_bins,
    unsigned int* const d_bins,
    unsigned int binidxs_per_thread,
    unsigned int bin_grid_size
) {
    extern __shared__ unsigned int sh_bins[];

    unsigned int absIdx = blockIdx.x*blockDim.x*binidxs_per_thread + threadIdx.x;
    unsigned int bDim   = blockDim.x;
    unsigned int idx    = threadIdx.x;
    unsigned int k;

    for (unsigned int i = 0; i < bin_grid_size; i++) {
        k = idx + i*bDim;
        if (k >= n_bins)
            break;
        sh_bins[k] = 0;
    }
    __syncthreads();

    unsigned int bin;
    for (unsigned int i = 0; i < binidxs_per_thread; i++) {
        k = absIdx + i*bDim;
        if (k >= length)
            break;
        bin = d_bin_idxs[k];
        atomicAdd(&sh_bins[bin], 1);
    }
    __syncthreads();

    for (unsigned int i = 0; i < bin_grid_size; i++) {
        k = idx + i*bDim;
        if (k >= n_bins)
            return;
        atomicAdd(&d_bins[k], sh_bins[k]);
    }
}


// -------------------- GPU Parallel Histogram (thrust) --------------------
void thrustHistogramWithBinIdxs(
    const float* const d_in,
    const unsigned int length,
    const unsigned int n_bins,
    unsigned int* const d_bins,
    unsigned int* const d_bin_idxs
) {
    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, 0) );
    unsigned int maxRunningThreads = deviceProp.multiProcessorCount*deviceProp.maxThreadsPerMultiProcessor;

    checkCudaErrors(cudaMemset(d_bins, 0, n_bins*sizeof(unsigned int)));

    float minimum, maximum;
    thrustGPUfindMinMaxFloat(d_in, length, minimum, maximum);
    float oneOverBinWidth = (1.0F*n_bins)/(maximum - minimum);

    auto d_lmbd = [d_in, minimum, n_bins, oneOverBinWidth, d_bin_idxs] __device__ (unsigned int i) {
        unsigned int bin = (d_in[i] - minimum)*oneOverBinWidth;
        d_bin_idxs[i] = min(bin, n_bins - 1);
    };
    thrust::counting_iterator<unsigned int> k(0U);
    thrust::for_each(thrust::device, k, k+length, d_lmbd);

    dim3 blockDim(1024, 1, 1);
    unsigned int binidxsPerThread = ui_ceilDiv(length, maxRunningThreads);
    unsigned int binGridSize = ui_ceilDiv(n_bins, blockDim.x);
    unsigned int gridX = ui_ceilDiv(length, binidxsPerThread*blockDim.x);
    dim3 gridDim(gridX, 1, 1);
    kernelMakeHistogram<<<gridDim, blockDim, n_bins*sizeof(unsigned int)>>>(d_bin_idxs, length, n_bins, d_bins, binidxsPerThread, binGridSize);
}


// -------------------- GPU Parallel Histogram --------------------
__global__ void kernelComputeBinidx(
    const float* const d_in,
    unsigned int length,
    unsigned int n_bins,
    const float minimum,
    const float oneOverBinWidth,
    unsigned int* const d_bin_idxs
) {
    unsigned int absIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (absIdx >= length)
        return;

    unsigned int bin = (d_in[absIdx] - minimum)*oneOverBinWidth;
    d_bin_idxs[absIdx] = min( bin, n_bins - 1 );
}


void parallelHistogramWithBinIdxs(
    const float* const d_in,
    unsigned int length,
    unsigned int n_bins,
    unsigned int* const d_bins,
    unsigned int* const d_bin_idxs
) {
    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, 0) );
    unsigned int maxRunningThreads = deviceProp.multiProcessorCount*deviceProp.maxThreadsPerMultiProcessor;

    memsetZero(d_bins, n_bins*sizeof(unsigned int));

    float min, max;
    parallelGPUfindMinMaxFloat(d_in, length, min, max);
    float oneOverBinWidth = (1.0F*n_bins)/(max - min);

    dim3 blockDim(1024, 1, 1);
    unsigned int gridX = ui_ceilDiv(length, blockDim.x);
    dim3 gridDim(gridX, 1, 1);
    kernelComputeBinidx<<<gridDim, blockDim>>>(d_in, length, n_bins, min, oneOverBinWidth, d_bin_idxs);

    unsigned int binidxsPerThread = ui_ceilDiv(length, maxRunningThreads);
    unsigned int binGridSize = ui_ceilDiv(n_bins, blockDim.x);
    gridX = ui_ceilDiv(length, binidxsPerThread*blockDim.x);
    gridDim.x = gridX;
    kernelMakeHistogram<<<gridDim, blockDim, n_bins*sizeof(unsigned int)>>>(d_bin_idxs, length, n_bins, d_bins, binidxsPerThread, binGridSize);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}
