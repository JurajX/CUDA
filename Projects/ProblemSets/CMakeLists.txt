cmake_minimum_required(VERSION 3.16)

project( HistogramTest LANGUAGES CUDA CXX )

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_STANDARD_REQUIRED TRUE )

set( CMAKE_CUDA_STANDARD 14 )
set( CMAKE_CUDA_STANDARD_REQUIRED TRUE )

if( NOT HW )
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries/HW"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries/HW/build")
endif( NOT HW )







add_executable( HomeWork main.cpp )

target_link_libraries( HomeWork PRIVATE lib_hw )



# set custom linker (nvcc)
set( CMAKE_LINKER "/usr/bin/nvcc" )
# build the executable linker with the custom linker
set( CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_LINKER>  <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>" )
# originally the linker is build as follows
# set( CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_CXX_COMPILER>  <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>" )
