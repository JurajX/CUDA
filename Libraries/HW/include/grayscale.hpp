#ifndef KERNEL_HPP__
#define KERNEL_HPP__



void serialBGRtoGreyscale(
    const unsigned char* const h_bgrDataPtr,
    unsigned int rows,
    unsigned int cols,
    unsigned char* const h_greyDataPtr
);

void parallelBGRtoGreyscale(
    const unsigned char* const h_bgrDataPtr,
    unsigned int rows,
    unsigned int cols,
    unsigned char* const h_greyDataPtr
);



#endif
