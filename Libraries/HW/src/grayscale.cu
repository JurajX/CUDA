#include "grayscale.hpp"

#include <cuda.h>
#include "checkCudaErrors.hpp"
#include "cudaMemory.hpp"


void serialBGRtoGreyscale(
    const unsigned char* const h_bgrDataPtr,
    unsigned int rows,
    unsigned int cols,
    unsigned char* const h_greyDataPtr
) {
    for (unsigned int i = 0; i < rows*cols; i++) {
        // grayPixel = .114f * B + .587f * G + .299f * R
        unsigned char grayPixel = (unsigned char)(0.114F*h_bgrDataPtr[3*i] + 0.587F*h_bgrDataPtr[3*i + 1] + 0.299F*h_bgrDataPtr[3*i + 2]);
        h_greyDataPtr[i] = grayPixel;
    }
}


__global__ void kernelBGRtoGreyscale(
    const uchar3* const d_bgrDataPtr,
    unsigned int rows,
    unsigned int cols,
    unsigned char* const d_greyDataPtr
) {
    unsigned int imgX = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int imgY = blockIdx.y*blockDim.y + threadIdx.y;
    if (imgX >= cols || imgY >= rows) {
        return;
    }
    unsigned int index = imgY*cols + imgX;
    uchar3 bgrPixel = d_bgrDataPtr[index];
    // grayPixel = .114f * B + .587f * G + .299f * R
    unsigned char grayPixel = (unsigned char)(0.114F*bgrPixel.x + 0.587F*bgrPixel.y + 0.299F*bgrPixel.z);
    d_greyDataPtr[index] = grayPixel;
}

void parallelBGRtoGreyscale(
    const unsigned char* const h_bgrDataPtr,
    unsigned int rows,
    unsigned int cols,
    unsigned char* const h_greyDataPtr
) {
    unsigned char * d_bgrDataPtr, * d_greyDataPtr;
    allocCudaMem((void**) &d_bgrDataPtr, 3*rows*cols*sizeof(unsigned char));            // gpuMemFree((void**) &d_bgrDataPtr);
    allocCudaMem((void**) &d_greyDataPtr,  rows*cols*sizeof(unsigned char));            // gpuMemFree((void**) &d_greyDataPtr);
    memcpyCPUtoGPU((void*) h_bgrDataPtr, (void*) d_bgrDataPtr, 3*rows*cols*sizeof(unsigned char));

    const dim3 blockDim(32, 32, 1);                         // maximum number of threads per block is 1024 = 32^2
    unsigned int gridX = (cols - 1)/(blockDim.x) + 1;       // ceil( cols/(blockDim.x) )
    unsigned int gridY = (rows - 1)/(blockDim.y) + 1;       // ceil( rows/(blockDim.y) )
    const dim3 gridDim(gridX, gridY, 1);

    kernelBGRtoGreyscale<<<gridDim, blockDim>>>((uchar3*)d_bgrDataPtr, rows, cols, d_greyDataPtr);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    memcpyGPUtoCPU((void*) d_greyDataPtr, (void*) h_greyDataPtr, rows*cols*sizeof(unsigned char));

    gpuMemFree((void**) &d_bgrDataPtr);
    gpuMemFree((void**) &d_greyDataPtr);
}
