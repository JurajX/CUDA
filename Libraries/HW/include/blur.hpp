#ifndef BLUR_HPP__
#define BLUR_HPP__



void serialGaussianBlurr(
    unsigned char* const h_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    const float* const h_filterPtr,
    const unsigned int filterWidth,
    unsigned char* const h_blurDataPtr
);

void parallelGaussianBlurr(
    unsigned char* const h_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    const float* const h_filterPtr,
    const unsigned int filterWidth,
    unsigned char* const h_blurDataPtr
);



#endif
