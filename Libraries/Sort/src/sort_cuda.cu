#include "sort.hpp"

#include "checkCudaErrors.hpp"
#include "cudaMemory.hpp"
#include "functions.hpp"

#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <limits>
#include <utility>
#include "scan.hpp"
#include "reduce.hpp"


// -------------------- GPU Parallel Sort (thrust) --------------------
void thrustGPUsort(
    unsigned int* const d_in,
    const unsigned int length
) {
    thrust::sort(thrust::device, d_in, d_in + length);
}

void thrustGPUsort(
    std::pair<unsigned int, unsigned int>* const d_in,
    const unsigned int length
) {
    thrust::sort(
        thrust::device,
        (thrust::pair<unsigned int, unsigned int>*) d_in,
        (thrust::pair<unsigned int, unsigned int>*) d_in + length
    );
}


// -------------------- GPU Parallel Sort (Bitonic Sort & Merge Sort) --------------------

//  ⇩ memory address;                            (⇩ ⇩ ⇩ ⇩) splitter network                       (⇩ ⇩ ⇩ ⇩) half_cleaner network
//  0 -----◯--------------◯------◯----------------◯----------◯------◯-----------◯------------------◯----------◯------◯---...
// idx=1-> |              |      | <-idx=1        |          |      |           |                  |          |      |   ...
//  1 -----◯----------------◯----◯------------------◯----------◯----◯-------------◯------------------◯----------◯----◯---...
//                idx=1-> | | <-idx=2             | |        | |                | |                | |        | |        ...
//  2 -----◯----------------◯----◯--------------------◯------◯------◯---------------◯------------------◯------◯------◯---...
// idx=2-> |              |      | <-idx=2        | | |        |    |           | | |              | | |        |    |   ...
//  3 -----◯--------------◯------◯----------------------◯------◯----◯-----------------◯------------------◯------◯----◯---...
//                                                | | | |                       | | | |            | | | |               ...
//  4 -----◯--------------◯------◯----------------------◯----◯------◯-------------------◯----------◯----------◯------◯---...
// idx=3-> |              |      | <-idx=3        | | |      |      |           | | | | |            | | |    |      |   ...
//  5 -----◯----------------◯----◯--------------------◯--------◯----◯---------------------◯----------◯----------◯----◯---...
//                idx=3-> | | <-idx=4             | |        | |                | | | | | |            | |    | |        ...
//  6 -----◯----------------◯----◯------------------◯--------◯------◯-----------------------◯----------◯------◯------◯---...
// idx=4-> |              |      | <-idx=4        |            |    |           | | | | | | |            |      |    |   ...
//  7 -----◯--------------◯------◯----------------◯------------◯----◯-------------------------◯----------◯------◯----◯---...
//  ⋮                                                                           ⋮ ⋮ ⋮ ⋮ ⋮ ⋮ ⋮ ⋮
//  i: |<- 1 ->|      |<----- 2 ---->|        |<------------ 4 -------->|   |<-------------------- 8 ------------------->|
//  j:     x              xxx   <1>               xxxxxxx   <-2->  <1>          xxxxxxxxxxxxxxx   <-- 4 -->  <-2->  <1>
// i represents number of threads needed to sort chunck of 2^(i+1) elements


template <typename T>
__device__ __forceinline__ void splitterKeys(
    T* sh_keys,
    unsigned int idx,
    unsigned int i
) {
    unsigned int mask     = i - 1;
    unsigned int lsbIdx   = (idx&(i - 1));                      // first log_2(i) least significant bits from idx
    unsigned int address1 = ((idx - lsbIdx) << 1) + lsbIdx;     // move all bits of idx with position > log_2(i) by one to the left
    unsigned int address2 = address1^(mask + i);                // flip all bits <= log_2(i)
    if ( sh_keys[address1] > sh_keys[address2] )
        thrust::swap(sh_keys[address1], sh_keys[address2]);
}

template <typename K, typename V>
__device__ __forceinline__ void splitterKeyValues(
    K* sh_keys,
    V* sh_vals,
    unsigned int idx,
    unsigned int i
) {
    unsigned int mask     = i - 1;
    unsigned int lsbIdx   = (idx&(i - 1));                      // first log_2(i) least significant bits from idx
    unsigned int address1 = ((idx - lsbIdx) << 1) + lsbIdx;     // move all bits of idx with position > log_2(i) by one to the left
    unsigned int address2 = address1^(mask + i);                // flip all bits <= log_2(i)
    if ( sh_keys[address1] > sh_keys[address2] ) {
        thrust::swap(sh_keys[address1], sh_keys[address2]);
        thrust::swap(sh_vals[address1], sh_vals[address2]);
    }
}


template <typename T>
__device__ __forceinline__ void halfCleanerKeys(
    T* sh_keys,
    unsigned int idx,
    unsigned int j
) {
    unsigned int lsbIdx   = (idx&(j - 1));                      // first log_2(j) least significant bits from idx
    unsigned int address1 = ((idx - lsbIdx) << 1) + lsbIdx;     // move all bits of idx with position > log_2(j) by one to the left
    unsigned int address2 = address1 + j;
    if ( sh_keys[address1] > sh_keys[address2] )
        thrust::swap(sh_keys[address1], sh_keys[address2]);
}

template <typename K, typename V>
__device__ __forceinline__ void halfCleanerKeyValues(
    K* sh_keys,
    V* sh_vals,
    unsigned int idx,
    unsigned int j
) {
    unsigned int lsbIdx   = (idx&(j - 1));                      // first log_2(j) least significant bits from idx
    unsigned int address1 = ((idx - lsbIdx) << 1) + lsbIdx;     // move all bits of idx with position > log_2(j) by one to the left
    unsigned int address2 = address1 + j;
    if ( sh_keys[address1] > sh_keys[address2] ) {
        thrust::swap(sh_keys[address1], sh_keys[address2]);
        thrust::swap(sh_vals[address1], sh_vals[address2]);
    }
}


template <typename T>
__global__ void kernelBitonicSort2048(
    T* const d_in,
    const unsigned int length,
    const T T_max
) {
    // this is needed for dynamically allocated shared memeory, else one will have name conflicts
    extern __shared__ __align__(sizeof(T)) unsigned char sh_mem[];
    T* sh_data = reinterpret_cast<T*>(sh_mem);

    unsigned int absIdx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idx    = threadIdx.x;
    unsigned int bDim = blockDim.x;

    // load data and perform the first comparison swap (i.e. i=1)
    if (2*absIdx + 1 < length) {
        sh_data[2*idx    ] = min( d_in[2*absIdx], d_in[2*absIdx + 1] );
        sh_data[2*idx + 1] = max( d_in[2*absIdx], d_in[2*absIdx + 1] );
    } else if (2*absIdx == length - 1) {
        sh_data[2*idx    ] = d_in[2*absIdx];
        sh_data[2*idx + 1] = T_max;
    } else {
        sh_data[2*idx    ] = T_max;
        sh_data[2*idx + 1] = T_max;
    }
    __syncthreads();

    unsigned int i, j;
    for (i = 2; i <= warpSize; i <<= 1) {          // warps are synchronised
        splitterKeys(sh_data, idx, i);
        for (j = i>>1; j > 0; j >>= 1) {            // warps are synchronised
            halfCleanerKeys(sh_data, idx, j);
        }
    }
    __syncthreads();
    for ( ; i <= bDim; i <<= 1) {
        splitterKeys(sh_data, idx, i);
        __syncthreads();
        for (j = i>>1; j > warpSize; j >>= 1) {
            halfCleanerKeys(sh_data, idx, j);
            __syncthreads();
        }
        for ( ; j > 0; j >>= 1) {                   // warps are synchronised
            halfCleanerKeys(sh_data, idx, j);
        }
        __syncthreads();
    }

    if (2*absIdx < length)
        d_in[2*absIdx] = sh_data[2*idx];
    if (2*absIdx + 1 < length)
        d_in[2*absIdx + 1] = sh_data[2*idx + 1];
}


template <typename T>
__device__ __forceinline__ unsigned int getPosition(
    const T element,
    const T* const d_data,
    unsigned int first,
    unsigned int last,
    bool equal
) {
    unsigned int mid = first + (last - first)/2;
    while (mid != first) {
        if (element < d_data[mid])
            last = mid;
        else
            first = mid;
        mid = first + (last - first)/2;
    }
    if (equal)
        return (element <= d_data[first])? first : last;
    else
        return (element <  d_data[first])? first : last;
}


template <typename T>
__global__ void kernelParallelMerge(
    const T* const d_in,
    const unsigned int length,
    const unsigned int len_sorted_chunk,
    const unsigned int exponent,
    T* const d_out
) {
    unsigned int absIdx = blockIdx.x*blockDim.x + threadIdx.x;

    if (absIdx >= length)
        return;

    unsigned int chunckIdx   = absIdx>>exponent;
    unsigned int chunckFirst = chunckIdx<<exponent;
    unsigned int mergedFirst, searchFirst, searchLast, newPosition;
    bool equal;
    if ((chunckIdx&1) == 0) {
        mergedFirst = chunckFirst;
        searchFirst = chunckFirst + len_sorted_chunk;
        searchLast  = min(searchFirst + len_sorted_chunk, length);
        equal = false;
    } else {
        mergedFirst = chunckFirst - len_sorted_chunk;
        searchFirst = mergedFirst;
        searchLast  = chunckFirst;
        equal = true;
    }

    if (searchFirst >= length)
        return;

    newPosition  = absIdx - chunckFirst;
    newPosition += getPosition(d_in[absIdx], d_in, searchFirst, searchLast, equal) - searchFirst;
    newPosition += mergedFirst;

    d_out[newPosition] = d_in[absIdx];
}


template <typename T>
void pGPUsort(
    T* d_in,
    const unsigned int length,
    const T T_max
) {
    dim3 blockDim(1024, 1, 1);
    unsigned int gridX = ui_ceilDiv(length, 2*blockDim.x);
    dim3 gridDim(gridX, 1, 1);
    kernelBitonicSort2048<<<gridDim, blockDim, 2*blockDim.x*sizeof(T)>>>(d_in, length, T_max);

    gridX = ui_ceilDiv(length, blockDim.x);
    gridDim.x = gridX;

    T* d_inter;
    allocCudaMem((void**) &d_inter, length*sizeof(T));     // gpuMemFree((void**) &d_inter);

    T * d_1 = d_in, * d_2 = d_inter;
    unsigned int exponent = 11;         // 2^11 = 2048
    for (unsigned int lenSortedChunk = 2048; lenSortedChunk < length; lenSortedChunk <<= 1) {
        kernelParallelMerge<<<gridDim, blockDim>>>(d_1, length, lenSortedChunk, exponent, d_2);
        std::swap(d_1, d_2);
        exponent++;
    }
    memcpyGPUtoGPU((void*) d_1, (void*) d_in, length*sizeof(T));

    gpuMemFree((void**) &d_inter);
}


void parallelGPUsort(
    unsigned int* const d_in,
    const unsigned int length
) {
    unsigned int T_max = std::numeric_limits<unsigned int>::max();
    pGPUsort(d_in, length, T_max);
}

void parallelGPUsort(
    std::pair<unsigned int, unsigned int>* const d_in,
    const unsigned int length
) {
    unsigned int UI_MAX = std::numeric_limits<unsigned int>::max();
    auto T_max = thrust::pair<unsigned int, unsigned int>(UI_MAX, UI_MAX);
    pGPUsort( (thrust::pair<unsigned int, unsigned int>*) d_in, length, T_max );
}


// -------------------- GPU Parallel Radix Sort --------------------
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

template <typename T>
__global__ void kernelSortTile(
    T* const d_in,
    const unsigned int length,
    const T T_max,
    const unsigned int r_shift,
    const unsigned int full_mask,
    const unsigned int n_bins,
    unsigned int* const d_hist,
    unsigned int* const d_offsets
) {
    unsigned int absIdx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idx    = threadIdx.x;
    unsigned int bDim   = blockDim.x;
    unsigned int hbDim  = bDim>>1;
    unsigned int gDim   = gridDim.x;

    // this is needed for dynamically allocated shared memeory, else one will have name conflicts
    extern __shared__ __align__(sizeof(T)) unsigned char sh_mem[];
    T* sh_data = reinterpret_cast<T*>(sh_mem);

    T*            sh_vals =                  sh_data;       // length = bDim
    unsigned int* sh_keys = (unsigned int*) &sh_vals[bDim]; // length = bDim
    unsigned int* sh_hist = (unsigned int*) &sh_keys[bDim]; // length = n_bins

    if (idx < n_bins)                               // NOTE: this works only in case when (n_bins < bDim), else sh_hist will contain rubbish
        sh_hist[idx] = 0U;

    unsigned int key;
    T val;
    if (absIdx < length)
        val = d_in[absIdx];
    else
        val = T_max;

    key = getKey(val);
    key = (key >> r_shift) & full_mask;

    __syncthreads();
    atomicAdd(&sh_hist[key], 1);

    key = (key << 16) + idx;
    sh_keys[idx] = key;
    sh_vals[idx] = val;

    if(idx >= hbDim)
        return;
    __syncthreads();

    // Bitonic sort
    unsigned int i, j;
    for (i = 1; i <= warpSize; i <<= 1) {       // warps are synchronised
        splitterKeyValues(sh_keys, sh_vals, idx, i);
        for (j = i>>1; j > 0; j >>= 1) {        // warps are synchronised
            halfCleanerKeyValues(sh_keys, sh_vals, idx, j);
        }
    }
    __syncthreads();
    for ( ; i <= hbDim; i <<= 1) {
        splitterKeyValues(sh_keys, sh_vals, idx, i);
        __syncthreads();
        for (j = i>>1; j > warpSize; j >>= 1) {
            halfCleanerKeyValues(sh_keys, sh_vals, idx, j);
            __syncthreads();
        }
        for ( ; j > 0; j >>= 1) {               // warps are synchronised
            halfCleanerKeyValues(sh_keys, sh_vals, idx, j);
        }
        __syncthreads();
    }

    // Copy data to global memory
    if (absIdx + hbDim < length)
        d_in[absIdx + hbDim] = sh_vals[idx + hbDim];
    if (absIdx < length)
        d_in[absIdx] = sh_vals[idx];

    // NOTE: this works only in case when (n_bins < warpSize < bDim/2)
    if (idx >= n_bins)
        return;

    d_hist[blockIdx.x + idx*gDim] = sh_hist[idx];
    // scan [sh_hist, sh_hist + n_bin)
    for (unsigned int i = 1; i < n_bins-1; i <<= 1) {
        if (idx >= i) {
            sh_hist[idx] += sh_hist[idx - i];
        }
    }
    d_offsets[blockIdx.x*n_bins + idx] = sh_hist[idx];
}


template <typename T>
__global__ void kernelMoveElements(
    T* const d_in,
    const unsigned int length,
    const unsigned int r_shift,
    const unsigned int full_mask,
    const unsigned int n_bins,
    const unsigned int* d_scan,
    const unsigned int* const d_offsets,
    T* const d_out
) {
    // this is needed for dynamically allocated shared memeory, else one will have name conflicts
    extern __shared__ __align__(sizeof(unsigned int)) unsigned char sh_mem[];
    unsigned int* sh_hist = reinterpret_cast<unsigned int*>(sh_mem);

    unsigned int absIdx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int idx    = threadIdx.x;
    unsigned int gDim   = gridDim.x;

    if (absIdx >= length)
        return;

    if (idx == 0)
        sh_hist[0] = 0;

    if (idx < n_bins-1) {
    sh_hist[idx + 1] = d_offsets[blockIdx.x*n_bins + idx];
    }

    T            val = d_in[absIdx];
    unsigned int key = getKey(val);
    unsigned int bin = (key >> r_shift) & full_mask;
    __syncthreads();
    unsigned int newPosition = d_scan[blockIdx.x + bin*gDim] + idx - sh_hist[bin];
    d_out[newPosition] = val;
}

template <typename T>
unsigned int getAllBits(
    T* const d_in,
    const unsigned int length
) {
    unsigned int allBits;
    if ( sizeof(T) == sizeof(unsigned int) )
        allBits = parallelGPUreduceBitOr( (unsigned int*)d_in, length );
    else
        allBits = parallelGPUreduceBitOr( (std::pair<unsigned int, unsigned int>*)d_in, length );
    return allBits;
}

template <typename T>
void pGPUradixsort(
    T* const d_in,
    const unsigned int length,
    const T T_max
) {
    dim3 blockDim(128, 1, 1);
    unsigned int gridX = ui_ceilDiv(length, blockDim.x);
    dim3 gridDim(gridX, 1, 1);

    unsigned int n_bits = 5;                        // works only for n_bins < warpSize
    unsigned int n_bins = 1 << n_bits;              // 2^n_bins
    unsigned int fullMask = (1 << n_bits) - 1;
    unsigned int sh_len1 = (blockDim.x)*sizeof(T) + (blockDim.x + n_bins)*sizeof(unsigned int);
    unsigned int sh_len2 = n_bins*sizeof(unsigned int);

    unsigned int allBits = getAllBits(d_in, length);
    unsigned int bitpos = 0;
    while (allBits != 0) {
        bitpos   += 1;
        allBits >>= 1;
    }

    unsigned int * d_hist, * d_offsets;
    allocCudaMem((void**) &d_hist,    (gridDim.x*n_bins + 1)*sizeof(unsigned int)); // gpuMemFree((void**) &d_hist);
    allocCudaMem((void**) &d_offsets, (gridDim.x*n_bins    )*sizeof(unsigned int)); // gpuMemFree((void**) &d_offsets);

    T* d_tmp;
    allocCudaMem((void**) &d_tmp, length*sizeof(T));                     // gpuMemFree((void**) &d_tmp);

    T* d_1 = d_in, * d_2 = d_tmp;

    for (unsigned int r_shift = 0; r_shift < bitpos; r_shift += n_bits) {
        if (bitpos - r_shift < n_bits) {
            n_bits = bitpos - r_shift;
            n_bins = 1 << n_bits;
            fullMask = (1 << n_bits) - 1;
            sh_len1 = (blockDim.x)*sizeof(T) + (blockDim.x + n_bins)*sizeof(unsigned int);
            sh_len2 = n_bins*sizeof(unsigned int);
        }
        kernelSortTile<<<gridDim, blockDim, sh_len1>>>(d_1, length, T_max, r_shift, fullMask, n_bins, d_hist+1, d_offsets);
        parallelGPUscan(d_hist+1, n_bins*gridDim.x, d_hist+1);
        kernelMoveElements<<<gridDim, blockDim, sh_len2>>>(d_1, length, r_shift, fullMask, n_bins, d_hist, d_offsets, d_2);
        std::swap(d_1, d_2);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    if (d_1 != d_in)
        memcpyGPUtoGPU((void*) d_1, (void*) d_in, length*sizeof(T));

    gpuMemFree((void**) &d_hist);
    gpuMemFree((void**) &d_offsets);
    gpuMemFree((void**) &d_tmp);
}


void parallelGPUradixsort(
    unsigned int* const d_in,
    const unsigned int length
) {
    unsigned int UI_MAX = std::numeric_limits<unsigned int>::max();
    pGPUradixsort(d_in, length, UI_MAX);
}

void parallelGPUradixsort(
    std::pair<unsigned int, unsigned int>* const d_in,
    const unsigned int length
) {
    unsigned int UI_MAX = std::numeric_limits<unsigned int>::max();
    auto T_max = thrust::pair<unsigned int, unsigned int>(UI_MAX, UI_MAX);
    pGPUradixsort( (thrust::pair<unsigned int, unsigned int>*)d_in, length, T_max );
}
