#include "HW.hpp"

#include <iostream>
#include <string>
#include <filesystem>

#include "timerCPU.hpp"
#include "timerGPU.hpp"
#include "image.hpp"
#include "grayscale.hpp"
#include "blur.hpp"
#include "filter.hpp"
#include "hdr.hpp"
#include "red_eye_removal.hpp"


void HW1(
    const std::filesystem::path cwd,
    const std::string imgName
) {
    Image imageBRG = Image(cwd/imgName, 3);
    Image imageOut = Image(imageBRG.getRows(), imageBRG.getCols(), 1);

    CpuTimer ctimer;
    ctimer.start();
    serialBGRtoGreyscale(
        imageBRG.getDataPtr(),
        imageBRG.getRows(),
        imageBRG.getCols(),
        imageOut.getDataPtr()
    );
    ctimer.stop();
    std::cout << "The CPU BGRtoGreyscale function run in: " << ctimer.elapsed() << " ms." << std::endl;
    imageOut.saveImage(cwd/("CPU_"+imgName));

    GpuTimer gtimer;
    gtimer.start();
    parallelBGRtoGreyscale(
        imageBRG.getDataPtr(),
        imageBRG.getRows(),
        imageBRG.getCols(),
        imageOut.getDataPtr()
    );
    gtimer.stop();
    std::cout << "The GPU BGRtoGreyscale function run in: " << gtimer.elapsed() << " ms." << std::endl;
    imageOut.saveImage(cwd/("GPU_"+imgName));
}


void HW2(
    const std::filesystem::path cwd,
    const std::string imgName
) {
    Image imageBRG     = Image(cwd/imgName, 3);
    Image imageOut = Image(imageBRG.getRows(), imageBRG.getCols(), 3);
    Filter filter = Filter(9);

    CpuTimer ctimer;
    ctimer.start();
    serialGaussianBlurr(
        imageBRG.getDataPtr(),
        imageBRG.getRows(),
        imageBRG.getCols(),
        filter.getDataPtr(),
        filter.getDimX(),
        imageOut.getDataPtr()
    );
    ctimer.stop();
    std::cout << "The CPU GaussianBlurr function run in: " << ctimer.elapsed() << " ms." << std::endl;
    imageOut.saveImage(cwd/("CPU_"+imgName));

    GpuTimer gtimer;
    gtimer.start();
    parallelGaussianBlurr(
        imageBRG.getDataPtr(),
        imageBRG.getRows(),
        imageBRG.getCols(),
        filter.getDataPtr(),
        filter.getDimX(),
        imageOut.getDataPtr()
    );
    gtimer.stop();
    std::cout << "The GPU GaussianBlurr function run in: " << gtimer.elapsed() << " ms." << std::endl;
    imageOut.saveImage(cwd/("GPU_"+imgName));
}


void HW3(
    const std::filesystem::path cwd,
    const std::string imgName
) {
    Image imageBRG = Image(cwd/imgName, 3);
    Image imageOut = Image(imageBRG.getRows(), imageBRG.getCols(), 3);
    unsigned int nBins = 1024;

    CpuTimer ctimer;
    ctimer.start();
    serialHDR(
        imageBRG.getDataPtr(),
        imageBRG.getRows(),
        imageBRG.getCols(),
        nBins,
        imageOut.getDataPtr()
    );
    ctimer.stop();
    std::cout << "The CPU HDR function run in: " << ctimer.elapsed() << " ms." << std::endl;
    imageOut.saveImage(cwd/("CPU_"+imgName));

    GpuTimer gtimer;
    gtimer.start();
    parallelHDR(
        imageBRG.getDataPtr(),
        imageBRG.getRows(),
        imageBRG.getCols(),
        nBins,
        imageOut.getDataPtr()
    );
    gtimer.stop();
    std::cout << "The GPU HDR function run in: " << gtimer.elapsed() << " ms." << std::endl;
    imageOut.saveImage(cwd/("GPU_"+imgName));
}


void HW4(
    const std::filesystem::path cwd,
    const std::string imgName,
    const std::string imgNameRedEye
) {
    Image imageBRG    = Image(cwd/imgName,       3);
    Image imageRedEye = Image(cwd/imgNameRedEye, 3);
    Image imageOut = Image(imageBRG.getRows(), imageBRG.getCols(), 3);

    // CpuTimer ctimer;
    // ctimer.start();
    // serialRedEyeRemoval(
    //     imageBRG.getDataPtr(),
    //     imageBRG.getRows(),
    //     imageBRG.getCols(),
    //     imageRedEye.getDataPtr(),
    //     imageRedEye.getRows(),
    //     imageRedEye.getCols(),
    //     imageOut.getDataPtr()
    // );
    // ctimer.stop();
    // std::cout << "The CPU red eye removal function run in: " << ctimer.elapsed() << " ms." << std::endl;
    // imageOut.saveImage(cwd/("CPU_"+imgName));

    GpuTimer gtimer;
    gtimer.start();
    parallelRedEyeRemoval(
        imageBRG.getDataPtr(),
        imageBRG.getRows(),
        imageBRG.getCols(),
        imageRedEye.getDataPtr(),
        imageRedEye.getRows(),
        imageRedEye.getCols(),
        imageOut.getDataPtr()
    );
    gtimer.stop();
    std::cout << "The GPU red eye removal function run in: " << gtimer.elapsed() << " ms." << std::endl;
    imageOut.saveImage(cwd/("GPU_"+imgName));
}
