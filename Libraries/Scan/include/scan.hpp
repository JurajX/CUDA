#ifndef SCAN_HPP__
#define SCAN_HPP__



// cuda file
void thrustGPUscan(
    const unsigned int* const d_in,
    const unsigned int length,
    unsigned int* const d_out
);

void parallelGPUscan(
    unsigned int* const d_in,
    const unsigned int length,
    unsigned int* const d_out
);


// cpp file
void serialCPUscan(
    const unsigned int* const h_in,
    const unsigned int length,
    unsigned int* const h_out
);

void stdParallelCPUscan(
    const unsigned int* const h_in,
    const unsigned int length,
    unsigned int* const h_out
);



#endif
