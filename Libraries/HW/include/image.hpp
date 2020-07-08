#ifndef IMAGE_HPP__
#define IMAGE_HPP__



#include <filesystem>


class Image {
private:
    unsigned int   n_rows = 0;
    unsigned int   n_cols = 0;
    unsigned int   n_chan = 0;
    unsigned char* data_ptr = nullptr;

    void setRows(unsigned int rows)         { n_rows   = rows;     }
    void setCols(unsigned int cols)         { n_cols   = cols;     }
    void setChannels(unsigned int channels) { n_chan   = channels; }
    void setDataPtr(unsigned char* ptr)     { data_ptr = ptr;      }

public:
    Image() = delete;
    Image(const std::filesystem::path& path, unsigned int channels);
    Image(unsigned int rows, unsigned int cols, unsigned int channels);

    Image(const Image&);
    Image& operator=(const Image&);
    Image(Image&&);
    Image& operator=(Image&&);

    ~Image();

    unsigned int   getRows()     const { return n_rows;               }
    unsigned int   getCols()     const { return n_cols;               }
    unsigned int   getChannels() const { return n_chan;               }
    unsigned int   getSize()     const { return n_rows*n_cols*n_chan; }
    unsigned char* getDataPtr()  const { return data_ptr;             }

    void saveImage(const std::filesystem::path& path);
};



#endif
