#ifndef HDR_HPP__
#define HDR_HPP__



void serialHDR(
    const unsigned char* const h_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int n_bins,
    unsigned char* const h_hdrDataPtr
);

void parallelHDR(
    const unsigned char* const h_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int n_bins,
    unsigned char* const h_hdrDataPtr
);



#endif
