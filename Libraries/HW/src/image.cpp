#include "image.hpp"

#include <memory>
#include <utility>
#include <stdexcept>
#include <iostream>

#include <opencv2/core/mat.hpp>         // for cv::Mat
#include <opencv2/imgcodecs.hpp>        // for cv::imread, cv::imwrite, cv::IMREAD_COLOR, cv::IMREAD_GRAYSCALE
#include <opencv2/core/hal/interface.h> // for CV_8UC1, CV_8UC4


Image::Image(const std::filesystem::path& path, unsigned int channels) {
    cv::Mat img;
    if      (channels == 3)
        img = cv::imread(path.string(), cv::IMREAD_COLOR);
    else if (channels == 1)
        img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    else
        throw std::invalid_argument("The argument channels must have value either 3 or 1");

    if (img.empty())
        throw std::runtime_error("The loaded image " + path.string() + " is empty.");

    n_rows = img.rows;
    n_cols = img.cols;
    n_chan = channels;
    data_ptr = new unsigned char[this->getSize()];
    std::copy(img.ptr<unsigned char>(0), img.ptr<unsigned char>(0) + this->getSize(), data_ptr);
}

Image::Image(unsigned int rows, unsigned int cols, unsigned int channels):
        n_rows{ rows },
        n_cols{ cols },
        n_chan{ channels },
        data_ptr{ new unsigned char[this->getSize()] }
{}

Image::Image(const Image& img):
        n_rows{ img.getRows()     },
        n_cols{ img.getCols()     },
        n_chan{ img.getChannels() },
        data_ptr{ new unsigned char[img.getSize()] }
{
    std::uninitialized_copy(img.getDataPtr(), img.getDataPtr()+img.getSize(), data_ptr);
}

Image& Image::operator=(const Image & img) {
    if ( (this->getRows() != img.getRows()) || (this->getCols() != img.getCols()) ) {
        throw std::runtime_error("bad size in Image = ...");
    }
    std::copy(img.getDataPtr(), img.getDataPtr()+img.getSize(), data_ptr);
    return *this;
}

Image::Image(Image&& img):
        n_rows{ img.getRows()     },
        n_cols{ img.getCols()     },
        n_chan{ img.getChannels() },
        data_ptr{ img.getDataPtr() }
{
    img.setDataPtr(nullptr);
    img.setRows(0);
    img.setCols(0);
    img.setChannels(0);
}

Image& Image::operator=(Image&& img) {
    std::swap(n_rows,   img.n_rows);
    std::swap(n_cols,   img.n_cols);
    std::swap(n_chan,   img.n_chan);
    std::swap(data_ptr, img.data_ptr);
    return *this;
}

Image::~Image() {
    delete [] data_ptr;
}

void Image::saveImage(const std::filesystem::path& path) {
    cv::Mat img;
    if (this->getChannels() == 3)
        img = cv::Mat(this->getRows(), this->getCols(), CV_8UC3, (void*)this->getDataPtr() );
    else if (this->getChannels() == 1)
        img = cv::Mat(this->getRows(), this->getCols(), CV_8UC1, (void*)this->getDataPtr() );
    else
        throw std::invalid_argument("channels must have value either 3 or 1");

    try {
        cv::imwrite(path.string(), img);
        std::cout << "Image written to: " << path.string() << std::endl;
    }
    catch (const cv::Exception& ex) {
        std::cerr << "Could not save the image: " << path.string() << ".\n" << ex.what() << std::endl;
    }
}
