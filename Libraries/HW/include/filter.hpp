#ifndef FILTER_HPP__
#define FILTER_HPP__



class Filter {
private:
    unsigned int dim_x = 0;
    unsigned int dim_y = 0;
    float* data_ptr = nullptr;

    void setDimX(unsigned int dim) { dim_x = dim;    }
    void setDimY(unsigned int dim) { dim_y = dim;    }
    void setDataPtr(float* ptr)    { data_ptr = ptr; }

public:
    Filter() = delete;
    Filter(unsigned int size);

    Filter(const Filter&);
    Filter& operator=(const Filter&);
    Filter(Filter&&);
    Filter& operator=(Filter&&);

    ~Filter();

    unsigned int getDimX()    const { return dim_x;       }
    unsigned int getDimY()    const { return dim_y;       }
    unsigned int getSize()    const { return dim_x*dim_y; }
    float*       getDataPtr() const { return data_ptr;    }
};



#endif
