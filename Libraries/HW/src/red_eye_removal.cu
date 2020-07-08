#include "red_eye_removal.hpp"

#include "checkCudaErrors.hpp"
#include "cudaMemory.hpp"
#include "functions.hpp"

#include <cuda.h>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/pair.h>

#include <sort.hpp>


// ---------- Serial Red Eye Removal
// void serialRedEyeRemoval(
//     const unsigned char* const h_bgr_face_img,
//     const unsigned int rows,
//     const unsigned int cols,
//     const unsigned char* const h_red_eye_img,
//     const unsigned int eye_rows,
//     const unsigned int eye_cols,
//     unsigned char* const h_out_img
// ) {
//   // TODO
// }


// ---------- Parallel Red Eye Removal
template <typename T1, typename T2>
__device__ __host__ __forceinline__ T1 add_any3(
    const T1 &lhs,
    const T2 &rhs
) {
    return T1{ lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z };
}

template <typename T1, typename T2>
__device__ __host__ __forceinline__ float3 sub_flt3(
    const T1 &lhs,
    const T2 &rhs
) {
    return make_float3( (float)lhs.x - (float)rhs.x, (float)lhs.y - (float)rhs.y, (float)lhs.z - (float)rhs.z );
}

template <typename T1, typename T2>
__device__ __host__ __forceinline__ T1 mul_any3(
    const T1 &lhs,
    const T2 &rhs
) {
    return T1{ lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z };
}

template <typename T>
__device__ __host__ __forceinline__ float3 flt3_div_flt(
    const T &lhs,
    const float &rhs
) {
    return make_float3( (float)lhs.x/rhs, (float)lhs.y/rhs, (float)lhs.z/rhs );
}


// I will compute scalar product of deviations of X and Y and divide it by the norms, i.e. (Dev(X).Dev(Y))/(||Dev(X)||*||Dev(Y)||), where the deviation Dev(Z) = ( Z - E(Z) ) with E[] being expected value, '.' is a dot product, and ||Z|| is euclidean norm of Z.
__global__ void kernelKindaCrossCovariance(
    const uchar3* const d_bgr_face_img,
    const unsigned int rows,
    const unsigned int cols,
    const float3* const d_bgr_eye_img_dev,
    const unsigned int eye_rows,
    const unsigned int eye_cols,
    const unsigned int eye_length,
    const float3 bgr_eye_img_norm2,
    float* const d_cross_cov
) {
    unsigned int imgX = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int imgY = blockIdx.y*blockDim.y + threadIdx.y;
    if (imgX >= cols || imgY >= rows)
        return;

    uint3 bgrSum = make_uint3(0U, 0U, 0U);
    for (unsigned int windowIdx = 0; windowIdx < eye_length; windowIdx++) {
        int r = windowIdx/eye_cols - eye_rows/2;
        int c = windowIdx%eye_cols - eye_cols/2;
        r = min( max(r+(int)imgY, 0), (int)rows-1 );
        c = min( max(c+(int)imgX, 0), (int)cols-1 );
        unsigned int index = r*cols + c;

        bgrSum = add_any3(bgrSum, d_bgr_face_img[index]);
    }
    float3 bgrMean = flt3_div_flt(bgrSum, (float)eye_length);
    float3 sum_devFaceImg_mul_devEyeImg = make_float3(0.0F, 0.0F, 0.0F);
    float3 bgrFaceImgNorm2              = make_float3(0.0F, 0.0F, 0.0F);

    for (unsigned int windowIdx = 0; windowIdx < eye_length; windowIdx++) {
        int r = windowIdx/eye_cols - eye_rows/2;
        int c = windowIdx%eye_cols - eye_cols/2;
        r = min( max(r+(int)imgY, 0), (int)rows-1 );
        c = min( max(c+(int)imgX, 0), (int)cols-1 );
        unsigned int index = r*cols + c;
        float3 bgrFaceImgDeviation = sub_flt3(d_bgr_face_img[index], bgrMean);
        sum_devFaceImg_mul_devEyeImg = add_any3( sum_devFaceImg_mul_devEyeImg, mul_any3(bgrFaceImgDeviation, d_bgr_eye_img_dev[windowIdx]) );
        bgrFaceImgNorm2 = add_any3( bgrFaceImgNorm2, mul_any3(bgrFaceImgDeviation, bgrFaceImgDeviation) );
    }

    float3 denominator = mul_any3(bgr_eye_img_norm2, bgrFaceImgNorm2);
    float3 result = make_float3(0.0F, 0.0F, 0.0F);
    if (denominator.x != 0)
        result.x = sum_devFaceImg_mul_devEyeImg.x / sqrt(denominator.x);
    if (denominator.y != 0)
        result.y = sum_devFaceImg_mul_devEyeImg.y / sqrt(denominator.y);
    if (denominator.z != 0)
        result.z = sum_devFaceImg_mul_devEyeImg.z / sqrt(denominator.z);

    d_cross_cov[imgY*cols + imgX] = result.x * result.y * result.z * 1e6;
}


__global__ void kernelAdjustRed(
    const thrust::pair<unsigned int, unsigned int>* const d_corrPosPairs,
    unsigned int rows,
    unsigned int cols,
    unsigned int length,
    unsigned int adjust_rows,
    unsigned int adjust_cols,
    unsigned int adjust_length,
    uchar3* const d_bgr_face_img
) {
    unsigned int windowIdx    = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int nthMostCorr  = blockIdx.y;

    if (windowIdx >= adjust_length)
        return;

    unsigned int pixelIdx  = d_corrPosPairs[length - 1 - nthMostCorr].second;
    unsigned int imgX = pixelIdx%cols;
    unsigned int imgY = pixelIdx/cols;

    int r = windowIdx/adjust_cols - adjust_rows/2;
    int c = windowIdx%adjust_cols - adjust_cols/2;

    if (r*r + c*c >= adjust_length/4)
        return;

    r = min( max(r+(int)imgY, 0), (int)rows-1 );
    c = min( max(c+(int)imgX, 0), (int)cols-1 );
    unsigned int pxlToAdjust = r*cols + c;

    uchar3 pixel = d_bgr_face_img[pxlToAdjust];
    unsigned int bgAvg = (pixel.x + pixel.y)/2;
    pixel.z = (unsigned char)bgAvg;
    d_bgr_face_img[pxlToAdjust] = pixel;
}


void parallelRedEyeRemoval(
    const unsigned char* const h_bgr_face_img,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned char* const h_red_eye_img,
    const unsigned int eye_rows,
    const unsigned int eye_cols,
    unsigned char* const h_out_img
) {
    unsigned int length = rows*cols;
    unsigned int eyeLength = eye_rows*eye_cols;

    // copy data to device
    thrust::device_vector<uchar3> d_bgrFaceImg( (uchar3*)h_bgr_face_img,    (uchar3*)(h_bgr_face_img)    + length    );
    thrust::device_vector<uchar3> d_bgrEyeImg(  (uchar3*)h_red_eye_img, (uchar3*)(h_red_eye_img) + eyeLength );

    // compute deviation (y - mean(Y)) for every pixel and channel of EyeImg
    auto lmbd_any3_uint3  = [] __device__ ( const auto &e ) { return make_uint3( (unsigned int)(e.x), (unsigned int)(e.y), (unsigned int)(e.z) ); };
    auto lmbd_square_any3 = [] __device__ ( const auto &e ) { return decltype(e){e.x*e.x, e.y*e.y, e.z*e.z}; };
    auto lmbd_add_any3    = [] __device__ ( const auto &l, const auto &r ) { return decltype(l){ l.x + r.x, l.y + r.y, l.z + r.z }; };

    uint3 bgrRedEyeSum = thrust::transform_reduce(
        thrust::device,
        thrust::raw_pointer_cast(d_bgrEyeImg.data()),
        thrust::raw_pointer_cast(d_bgrEyeImg.data()) + eyeLength,
        lmbd_any3_uint3,
        make_uint3(0U, 0U, 0U),
        lmbd_add_any3
    );
    float3 bgrEyeImgMean = make_float3(
        (float)bgrRedEyeSum.x/(float)eyeLength,
        (float)bgrRedEyeSum.y/(float)eyeLength,
        (float)bgrRedEyeSum.z/(float)eyeLength
    );

    thrust::device_vector<float3> d_bgrEyeImgDeviation(eyeLength, make_float3(-bgrEyeImgMean.x, -bgrEyeImgMean.y, -bgrEyeImgMean.z));
    thrust::transform(
        thrust::device,
        thrust::raw_pointer_cast( d_bgrEyeImgDeviation.data() ),
        thrust::raw_pointer_cast( d_bgrEyeImgDeviation.data() ) + eyeLength,
        thrust::raw_pointer_cast( d_bgrEyeImg.data() ),
        thrust::raw_pointer_cast( d_bgrEyeImgDeviation.data() ),
        lmbd_add_any3
    );

    // compute normed of Dev(EyeImg) for each channel;
    float3 bgrEyeImgNorm2 = thrust::transform_reduce(
        thrust::device,
        thrust::raw_pointer_cast( d_bgrEyeImgDeviation.data() ),
        thrust::raw_pointer_cast( d_bgrEyeImgDeviation.data() ) + eyeLength,
        lmbd_square_any3,
        make_float3(0.0F, 0.0F, 0.0F),
        lmbd_add_any3
    );

    // compute "cross covariance" between a window centered at each pixel of the FaceImg with the EyeImg
    dim3 blockDim(32, 32, 1);
    dim3 gridDim( ui_ceilDiv(cols, blockDim.x), ui_ceilDiv(rows, blockDim.y), 1);

    thrust::device_vector<float> d_crossCorr(length);
    kernelKindaCrossCovariance<<<gridDim, blockDim, eyeLength*sizeof(float3)>>>(
        thrust::raw_pointer_cast( d_bgrFaceImg.data() ),
        rows,
        cols,
        thrust::raw_pointer_cast( d_bgrEyeImgDeviation.data() ),
        eye_rows,
        eye_cols,
        eyeLength,
        bgrEyeImgNorm2,
        thrust::raw_pointer_cast(d_crossCorr.data())
    );

    // subtract minimum elements from every element of the correlation array (i.e. make them positive)
    float min = *thrust::min_element(thrust::device, d_crossCorr.begin(), d_crossCorr.end());
    thrust::transform(
        d_crossCorr.begin(),
        d_crossCorr.end(),
        thrust::make_constant_iterator(-min),
        d_crossCorr.begin(),
        thrust::plus<float>()
    );

    // make an array of pairs, where first is correlation of a pixel and second is its position
    thrust::device_vector< thrust::pair<unsigned int, unsigned int> > d_corrPosPairs(length);
    thrust::transform(
        thrust::device,
        d_crossCorr.begin(),
        d_crossCorr.end(),
        thrust::make_counting_iterator(0U),
        d_corrPosPairs.begin(),
        [] __device__ (const auto &first, const auto &second) {
            return thrust::pair<unsigned int, unsigned int>(first, second);
        }
    );

    // sort the array according to first element
    parallelGPUradixsort( (std::pair<unsigned int, unsigned int>*) thrust::raw_pointer_cast( d_corrPosPairs.data() ), length );

    // for the 30 most correlated pixels with the EyeImg, center a window of dim=(neighborsToAdjust, neighborsToAdjust) at each one and adjust
    // the red channel for all the pixels within the window
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    unsigned int maxThreadsDim = *prop.maxThreadsDim/4;

    unsigned int neighborsToAdjust = 21;
    unsigned int lenToAdjust = neighborsToAdjust*neighborsToAdjust;
    unsigned int corrPxlToAdjust = 30;

    blockDim = dim3(std::min(lenToAdjust, maxThreadsDim), 1,  1);
    gridDim  = dim3(ui_ceilDiv(lenToAdjust, blockDim.x),  corrPxlToAdjust, 1);

    kernelAdjustRed<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast( d_corrPosPairs.data() ),
        rows,
        cols,
        length,
        neighborsToAdjust,
        neighborsToAdjust,
        lenToAdjust,
        thrust::raw_pointer_cast( d_bgrFaceImg.data() )
    );

    memcpyGPUtoCPU((void*) thrust::raw_pointer_cast( d_bgrFaceImg.data() ), (void*) h_out_img, 3*length*sizeof(unsigned char));
}
