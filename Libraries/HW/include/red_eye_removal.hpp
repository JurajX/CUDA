#ifndef RED_EYE_REMOVAL_HPP__
#define RED_EYE_REMOVAL_HPP__



// void serialRedEyeRemoval(
//     const unsigned char* const h_bgrDataPtr,
//     const unsigned int rows,
//     const unsigned int cols,
//     const unsigned char* const h_redEyeDataPtr,
//     const unsigned int eye_rows,
//     const unsigned int eye_cols,
//     unsigned char* const h_outDataPtr
// );

void parallelRedEyeRemoval(
    const unsigned char* const h_bgrDataPtr,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned char* const h_redEyeDataPtr,
    const unsigned int eye_rows,
    const unsigned int eye_cols,
    unsigned char* const h_outDataPtr
);



#endif
