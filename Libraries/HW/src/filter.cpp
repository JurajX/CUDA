#include "filter.hpp"

#include <memory>
#include <utility>
#include <stdexcept>
#include <cmath>


Filter::Filter(unsigned int size):
        dim_x{ size },
        dim_y{ size },
        data_ptr{ new float[this->getSize()] }
{
    const float sig = 2.0F;
    float sum = 0.0F;

    for (int i = 0; i< this->getSize(); i++) {
        int r = i/dim_y - dim_y/2;
        int c = i%dim_x - dim_x/2;
        float val = std::exp( -(r*r + c*c)/(2*sig*sig) );
        data_ptr[i] = val;
        sum += val;
    }
    for (int i = 0; i < this->getSize(); i++) {
        data_ptr[i] /= sum;
    }
}

Filter::Filter(const Filter& filter):
        dim_x{ filter.getDimX() },
        dim_y{ filter.getDimY() },
        data_ptr{ new float[filter.getSize()] }
{
    std::uninitialized_copy(filter.getDataPtr(), filter.getDataPtr()+filter.getSize(), data_ptr);
}

Filter& Filter::operator=(const Filter & filter) {
    if ( (this->getDimX() != filter.getDimX()) || (this->getDimY() != filter.getDimY()) ) {
        throw std::runtime_error("bad size in Filter = ...");
    }
    std::copy(filter.getDataPtr(), filter.getDataPtr()+filter.getSize(), data_ptr);
    return *this;
}

Filter::Filter(Filter&& filter):
        dim_x{ filter.getDimX() },
        dim_y{ filter.getDimY() },
        data_ptr{ filter.getDataPtr() }
{
    filter.setDataPtr(nullptr);
    filter.setDimX(0);
    filter.setDimY(0);
}

Filter& Filter::operator=(Filter&& filter) {
    std::swap(dim_x,    filter.dim_x);
    std::swap(dim_y,    filter.dim_y);
    std::swap(data_ptr, filter.data_ptr);
    return *this;
}

Filter::~Filter() {
    delete [] data_ptr;
}
