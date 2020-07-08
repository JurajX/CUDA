#include "blur.hpp"

#include <cuda.h>
#include "checkCudaErrors.hpp"
#include "cudaMemory.hpp"


void serialGaussianBlurr(
    unsigned char* const h_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    const float* const h_filterPtr,
    const unsigned int filterWidth,
    unsigned char* const h_blurredDataPtr
) {
    for (unsigned int j = 0; j < rows*cols; j++) {
        unsigned int imgX = j%cols;
        unsigned int imgY = j/cols;

        float pixelB = 0;
        float pixelG = 0;
        float pixelR = 0;
        for (unsigned int i = 0; i < filterWidth*filterWidth; i++) {
            int r = i/filterWidth - filterWidth/2;
            int c = i%filterWidth - filterWidth/2;
            r = min( max(r+(int)imgY, 0), (int)rows-1 );
            c = min( max(c+(int)imgX, 0), (int)cols-1 );
            pixelB += h_filterPtr[i]*h_bgrDataPtr[3*(r*cols + c)    ];
            pixelG += h_filterPtr[i]*h_bgrDataPtr[3*(r*cols + c) + 1];
            pixelR += h_filterPtr[i]*h_bgrDataPtr[3*(r*cols + c) + 2];
        }
        h_blurredDataPtr[3*j    ] = pixelB;
        h_blurredDataPtr[3*j + 1] = pixelG;
        h_blurredDataPtr[3*j + 2] = pixelR;
    }
}


__global__ void kernelSeparateChannels(
    const uchar3* const d_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    unsigned char* const d_bDataPtr,
    unsigned char* const d_gDataPtr,
    unsigned char* const d_rDataPtr
) {
    unsigned int imgX = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int imgY = blockIdx.y*blockDim.y + threadIdx.y;
    if (imgX >= cols || imgY >= rows) {
        return;
    }
    unsigned int index = imgY*cols + imgX;
    uchar3 bgrPixel = d_bgrDataPtr[index];
    d_bDataPtr[index] = bgrPixel.x;
    d_gDataPtr[index] = bgrPixel.y;
    d_rDataPtr[index] = bgrPixel.z;
}

__global__ void kernelRecombineChannels(
    const unsigned char* const d_bDataPtr,
    const unsigned char* const d_gDataPtr,
    const unsigned char* const d_rDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    uchar3* const d_bgrDataPtr
) {
    unsigned int imgX = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int imgY = blockIdx.y*blockDim.y + threadIdx.y;
    if (imgX >= cols || imgY >= rows) {
        return;
    }
    unsigned int index = imgY*cols + imgX;
    uchar3 bgrPixel = make_uchar3(d_bDataPtr[index], d_gDataPtr[index], d_rDataPtr[index]);
    d_bgrDataPtr[index] = bgrPixel;
}

__global__ void kernelBlur(
    const unsigned char* const d_dataPtr,
    const unsigned int rows,
    const unsigned int cols,
    const float* const d_filterPtr,
    const unsigned int filterWidth,
    unsigned char* const d_blurredDataPtr
) {
    unsigned int imgX = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int imgY = blockIdx.y*blockDim.y + threadIdx.y;
    if (imgX >= cols || imgY >= rows) {
        return;
    }
    float pixel = 0;
    for (unsigned int i = 0; i < filterWidth*filterWidth; i++) {
        int r = i/filterWidth - filterWidth/2;
        int c = i%filterWidth - filterWidth/2;
        r = min( max(r+(int)imgY, 0), (int)rows-1 );
        c = min( max(c+(int)imgX, 0), (int)cols-1 );
        pixel += d_filterPtr[i]*d_dataPtr[r*cols + c];
    }
    unsigned int index = imgY*cols + imgX;
    d_blurredDataPtr[index] = pixel;
}

void parallelGaussianBlurr(
    unsigned char* const h_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    const float* const h_filterPtr,
    const unsigned int filterWidth,
    unsigned char* const h_blurDataPtr
) {
    unsigned char* d_bgrDataPtr;
    allocCudaMem((void**) &d_bgrDataPtr, 3*rows*cols*sizeof(unsigned char));           // gpuMemFree((void**) &d_bgrDataPtr);
    memcpyCPUtoGPU((void*) h_bgrDataPtr, (void*) d_bgrDataPtr, 3*rows*cols*sizeof(unsigned char));

    float* d_filterPtr;
    allocCudaMem((void**) &d_filterPtr, filterWidth*filterWidth*sizeof(float));         // gpuMemFree((void**) &d_filterPtr);
    memcpyCPUtoGPU((void*) h_filterPtr, (void*) d_filterPtr, filterWidth*filterWidth*sizeof(float));

    unsigned char * d_bDataPtr, * d_gDataPtr, * d_rDataPtr;
    allocCudaMem((void**) &d_bDataPtr, rows*cols*sizeof(unsigned char));                // gpuMemFree((void**) &d_bDataPtr);
    allocCudaMem((void**) &d_gDataPtr, rows*cols*sizeof(unsigned char));                // gpuMemFree((void**) &d_gDataPtr);
    allocCudaMem((void**) &d_rDataPtr, rows*cols*sizeof(unsigned char));                // gpuMemFree((void**) &d_rDataPtr);

    unsigned char * d_bBlurDataPtr, * d_gBlurDataPtr, * d_rBlurDataPtr;
    allocCudaMem((void**) &d_bBlurDataPtr, rows*cols*sizeof(unsigned char));            // gpuMemFree((void**) &d_bBlurDataPtr);
    allocCudaMem((void**) &d_gBlurDataPtr, rows*cols*sizeof(unsigned char));            // gpuMemFree((void**) &d_gBlurDataPtr);
    allocCudaMem((void**) &d_rBlurDataPtr, rows*cols*sizeof(unsigned char));            // gpuMemFree((void**) &d_rBlurDataPtr);

    const dim3 blockDim(32, 32, 1);                         // maximum number of threads per block is 1024 = 32^2
    unsigned int gridX = (cols - 1)/(blockDim.x) + 1;       // ceil( cols/(blockDim.x) )
    unsigned int gridY = (rows - 1)/(blockDim.y) + 1;       // ceil( rows/(blockDim.y) )
    const dim3 gridDim(gridX, gridY, 1);

    kernelSeparateChannels<<<gridDim, blockDim>>>((uchar3*) d_bgrDataPtr, rows, cols, d_bDataPtr, d_gDataPtr, d_rDataPtr);

    kernelBlur<<<gridDim, blockDim>>>(d_bDataPtr, rows, cols, d_filterPtr, filterWidth, d_bBlurDataPtr);
    kernelBlur<<<gridDim, blockDim>>>(d_gDataPtr, rows, cols, d_filterPtr, filterWidth, d_gBlurDataPtr);
    kernelBlur<<<gridDim, blockDim>>>(d_rDataPtr, rows, cols, d_filterPtr, filterWidth, d_rBlurDataPtr);

    kernelRecombineChannels<<<gridDim, blockDim>>>(d_bBlurDataPtr, d_gBlurDataPtr, d_rBlurDataPtr, rows, cols, (uchar3*) d_bgrDataPtr);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    memcpyGPUtoCPU((void*) d_bgrDataPtr, (void*) h_blurDataPtr, 3*rows*cols*sizeof(unsigned char));

    gpuMemFree((void**) &d_bBlurDataPtr);
    gpuMemFree((void**) &d_gBlurDataPtr);
    gpuMemFree((void**) &d_rBlurDataPtr);
    gpuMemFree((void**) &d_bDataPtr);
    gpuMemFree((void**) &d_gDataPtr);
    gpuMemFree((void**) &d_rDataPtr);
    gpuMemFree((void**) &d_bgrDataPtr);
    gpuMemFree((void**) &d_filterPtr);
}
