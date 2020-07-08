#include "hdr.hpp"

#include <cuda.h>
#include "checkCudaErrors.hpp"
#include "cudaMemory.hpp"

#include "histogram.hpp"
#include "scan.hpp"

#include <cmath>


void serialBRGto_xylogY(
    const unsigned char* const h_bgrDataPtr,
    const unsigned int length,
    float* const x_ptr,
    float* const y_ptr,
    float* const logY_ptr
) {
    for (unsigned int i = 0; i < length; i++) {
        float b = (float)h_bgrDataPtr[3*i    ];
        float g = (float)h_bgrDataPtr[3*i + 1];
        float r = (float)h_bgrDataPtr[3*i + 2];

        float X = ( r * 0.4124F ) + ( g * 0.3576F ) + ( b * 0.1805F );
        float Y = ( r * 0.2126F ) + ( g * 0.7152F ) + ( b * 0.0722F );
        float Z = ( r * 0.0193F ) + ( g * 0.1192F ) + ( b * 0.9505F );

        float L = X + Y + Z;
        float x = X / L;
        float y = Y / L;

        x_ptr[i] = x;
        y_ptr[i] = y;
        logY_ptr[i] = std::log10( 0.0001F + Y );
    }
}

void serialNormaliseCDF(
    unsigned int* cdf,
    unsigned int length,
    float* normed_cdf
) {
    float norm_factor = 1.0F/cdf[length - 2];
    for (unsigned int i = 0; i < length; i++) {
        normed_cdf[i] = norm_factor*cdf[i];
    }
}

void serialTonemap(
    const unsigned int* const binIdxs,
    const float* const normedBinCDF,
    const unsigned int length,
    float* const Y_ptr
) {
    for (unsigned int i = 0; i < length; i++) {
        unsigned int bin = binIdxs[i];
        // exclusive scan... decrease bin number...
        Y_ptr[i] = (bin == 0)? 0 : normedBinCDF[bin-1];
    }
}

void serial_xyYtoBGR(
    float* const x_ptr,
    float* const y_ptr,
    float* const Y_ptr,
    unsigned int length,
    unsigned char* const h_hdrDataPtr
) {
    for (unsigned int i = 0; i < length; i++) {
        float x = x_ptr[i];
        float y = y_ptr[i];
        float Y = Y_ptr[i];
        float X = x * ( Y / y );
        float Z = ( 1 - x - y ) * ( Y / y );

        float r = ( (X *  3.2406F) + (Y * -1.5372F) + (Z * -0.4986F) )*255;
        float g = ( (X * -0.9689F) + (Y *  1.8758F) + (Z *  0.0415F) )*255;
        float b = ( (X *  0.0557F) + (Y * -0.2040F) + (Z *  1.0570F) )*255;

        unsigned char r_new = max( min( 255, (int)r), 0);
        unsigned char g_new = max( min( 255, (int)g), 0);
        unsigned char b_new = max( min( 255, (int)b), 0);

        h_hdrDataPtr[3*i    ] = b_new;
        h_hdrDataPtr[3*i + 1] = g_new;
        h_hdrDataPtr[3*i + 2] = r_new;
    }
}

void serialHDR(
    const unsigned char* const h_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int n_bins,
    unsigned char* const h_hdrDataPtr
) {
    float x_ptr[rows*cols];
    float y_ptr[rows*cols];
    float logY_ptr[rows*cols];
    unsigned int binIdxs[rows*cols];
    unsigned int bins[n_bins];
    float normedBinCDF[n_bins];
    unsigned int* binCDF = bins;
    float* Y_ptr = logY_ptr;

    serialBRGto_xylogY(h_bgrDataPtr, rows*cols, x_ptr, y_ptr, logY_ptr);
    serialHistogramWithBinIdxs(logY_ptr, rows*cols, n_bins, bins, binIdxs);
    serialCPUscan(bins, n_bins, binCDF);
    serialNormaliseCDF(binCDF, n_bins, normedBinCDF);
    serialTonemap(binIdxs, normedBinCDF, rows*cols, Y_ptr);
    serial_xyYtoBGR(x_ptr, y_ptr, Y_ptr, rows*cols, h_hdrDataPtr);
}


__global__ void kernelRGBextract_logY(
    const uchar3* const d_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    float* const d_logYDataPtr
) {
    unsigned int imgX = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int imgY = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int index = imgY*cols + imgX;
    if (imgX >= cols || imgY >= rows) {
        return;
    }

    uchar3 bgrPixel = d_bgrDataPtr[index];
    float b = (float)bgrPixel.x;
    float g = (float)bgrPixel.y;
    float r = (float)bgrPixel.z;

    float Y = ( r * 0.2126F ) + ( g * 0.7152F ) + ( b * 0.0722F );
    float log_Y = log10f( 0.0001F + Y );

    d_logYDataPtr[index] = log_Y;
}

__global__ void kernelBGRYtoBGR(
    const uchar3* const d_bgrDataPtr,
    const float* const d_YDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    uchar3* const d_hdrDataPtr
) {
    unsigned int imgX = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int imgY = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int index = imgY*cols + imgX;
    if (imgX >= cols || imgY >= rows) {
        return;
    }

    uchar3 bgrPixel = d_bgrDataPtr[index];
    float b = (float)bgrPixel.x;
    float g = (float)bgrPixel.y;
    float r = (float)bgrPixel.z;

    float X = ( r * 0.4124F ) + ( g * 0.3576F ) + ( b * 0.1805F );
    float Y = ( r * 0.2126F ) + ( g * 0.7152F ) + ( b * 0.0722F );
    float Z = ( r * 0.0193F ) + ( g * 0.1192F ) + ( b * 0.9505F );
    float L = X + Y + Z;
    float x = X / L;
    float y = Y / L;

    float Y_new = d_YDataPtr[index];
    float X_new = x * ( Y_new / y );
    float Z_new = ( 1 - x - y ) * ( Y_new / y );

    r = ( (X_new *  3.2406F) + (Y_new * -1.5372F) + (Z_new * -0.4986F) )*255;
    g = ( (X_new * -0.9689F) + (Y_new *  1.8758F) + (Z_new *  0.0415F) )*255;
    b = ( (X_new *  0.0557F) + (Y_new * -0.2040F) + (Z_new *  1.0570F) )*255;

    unsigned char r_new = max( min( 255, (int)r), 0);
    unsigned char g_new = max( min( 255, (int)g), 0);
    unsigned char b_new = max( min( 255, (int)b), 0);

    bgrPixel = make_uchar3(b_new, g_new, r_new);
    d_hdrDataPtr[index] = bgrPixel;
}

__global__ void kernelNormaliseCDF(
    const unsigned int* const d_CDF,
    float norm_factor,
    unsigned int length,
    float* const d_normedCDF
) {
    int absIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if (absIdx >= length)
        return;
    d_normedCDF[absIdx] = norm_factor*d_CDF[absIdx];
}

void parallelNormaliseCDF(
    unsigned int* d_CDF,
    unsigned int length,
    float* d_normedCDF
) {
    const dim3 blockDim(1024, 1, 1);
    unsigned int gridX = (length - 1)/blockDim.x + 1;
    const dim3 gridDim(gridX, 1, 1);

    unsigned int* d_lastCDF = d_CDF + (length - 2);
    unsigned int* lastCDF = new unsigned int;

    memcpyGPUtoCPU((void*) d_lastCDF, (void*) lastCDF, sizeof(float));
    float normFactor = 1.0F/(*lastCDF);

    kernelNormaliseCDF<<<gridDim, blockDim>>>(d_CDF, normFactor, length, d_normedCDF);
    delete lastCDF;
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

__global__ void kernelTonemap(
    const unsigned int* const d_binIdxs,
    const float* const d_normedBinCDF,
    const unsigned int rows,
    const unsigned int cols,
    float* const d_YDataPtr
) {
    unsigned int imgX = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int imgY = blockIdx.y*blockDim.y + threadIdx.y;
    if (imgX >= cols || imgY >= rows) {
        return;
    }
    unsigned int index = imgY*cols + imgX;

    unsigned int bin = d_binIdxs[index];
    // exclusive scan... decrease bin number...
    d_YDataPtr[index] = (bin == 0)? 0 : d_normedBinCDF[bin-1];
    // d_YDataPtr[index] = d_normedBinCDF[bin];
}

void parallelHDR(
    const unsigned char* const h_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int n_bins,
    unsigned char* const h_hdrDataPtr
) {
    unsigned char * d_bgrDataPtr, * d_hdrDataPtr;
    allocCudaMem((void**) &d_bgrDataPtr, 3*rows*cols*sizeof(unsigned char));    // gpuMemFree((void**) &d_bgrDataPtr);
    allocCudaMem((void**) &d_hdrDataPtr, 3*rows*cols*sizeof(unsigned char));    // gpuMemFree((void**) &d_hdrDataPtr);
    memcpyCPUtoGPU((void*) h_bgrDataPtr, (void*) d_bgrDataPtr, 3*rows*cols*sizeof(unsigned char));

    float* d_logYDataPtr, * d_normedBinCDF;
    allocCudaMem((void**) &d_logYDataPtr, rows*cols*sizeof(float));             // gpuMemFree((void**) &d_logYDataPtr);
    allocCudaMem((void**) &d_normedBinCDF,  n_bins*sizeof(float));              // gpuMemFree((void**) &d_normedBinCDF);
    float* d_YDataPtr = d_logYDataPtr;

    unsigned int * d_binIdxs, * d_bins, * d_binCDF;
    allocCudaMem((void**) &d_binIdxs, rows*cols*sizeof(unsigned int));          // gpuMemFree((void**) &d_binIdxs);
    allocCudaMem((void**) &d_bins,    n_bins*sizeof(unsigned int));             // gpuMemFree((void**) &d_bins);
    allocCudaMem((void**) &d_binCDF,  n_bins*sizeof(unsigned int));             // gpuMemFree((void**) &d_binCDF);

    const dim3 blockDim(32, 32, 1);                                             // maximum number of threads per block is 1024 = 32^2
    unsigned int gridX = (cols - 1)/(blockDim.x) + 1;                           // ceil( cols/(blockDim.x) )
    unsigned int gridY = (rows - 1)/(blockDim.y) + 1;                           // ceil( rows/(blockDim.y) )
    const dim3 gridDim(gridX, gridY, 1);

    kernelRGBextract_logY<<<gridDim, blockDim>>>((uchar3*)d_bgrDataPtr, rows, cols, d_logYDataPtr);
    parallelHistogramWithBinIdxs(d_logYDataPtr, rows*cols, n_bins, d_bins, d_binIdxs);
    parallelGPUscan(d_bins, n_bins, d_binCDF);
    parallelNormaliseCDF(d_binCDF, n_bins, d_normedBinCDF);
    kernelTonemap<<<gridDim, blockDim>>>(d_binIdxs, d_normedBinCDF, rows, cols, d_YDataPtr);
    kernelBGRYtoBGR<<<gridDim, blockDim>>>((uchar3*)d_bgrDataPtr, d_YDataPtr, rows, cols, (uchar3*)d_hdrDataPtr);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    memcpyGPUtoCPU((void*) d_hdrDataPtr, (void*) h_hdrDataPtr, 3*rows*cols*sizeof(unsigned char));

    gpuMemFree((void**) &d_bgrDataPtr);
    gpuMemFree((void**) &d_hdrDataPtr);
    gpuMemFree((void**) &d_logYDataPtr);
    gpuMemFree((void**) &d_normedBinCDF);
    gpuMemFree((void**) &d_binIdxs);
    gpuMemFree((void**) &d_bins);
    gpuMemFree((void**) &d_binCDF);
}
