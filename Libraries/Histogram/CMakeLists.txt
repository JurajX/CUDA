# ====================== Histogram
set( Histogram ON CACHE BOOL "Flag that Histogram Lib is included" )


add_library( lib_histogram STATIC
    ./include/histogram.hpp
    ./src/histogram_cuda.cu
    ./src/histogram_std.cpp
)

if( NOT SupportingFiles )
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../SupportingFiles"
        "${CMAKE_CURRENT_SOURCE_DIR}/../SupportingFiles/build")
endif( NOT SupportingFiles )

if( NOT Reduce )
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../Reduce"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Reduce/build")
endif( NOT Reduce )







set_target_properties( lib_histogram PROPERTIES CUDA_SEPERABLE_COMPILATION ON )
set_source_files_properties(./src/histogram_cuda.cu PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")

find_package( PkgConfig REQUIRED )                                  # the `pkg_check_modules` function is created with this call
pkg_check_modules( tbb REQUIRED IMPORTED_TARGET tbb )               # these calls create special `PkgConfig::<MODULE>` variables
target_link_libraries( lib_histogram INTERFACE PkgConfig::tbb )


target_link_libraries( lib_histogram PRIVATE lib_supporting_files_cuda )
target_link_libraries( lib_histogram PRIVATE lib_supporting_files )
target_link_libraries( lib_histogram PRIVATE lib_reduce )


target_include_directories( lib_histogram PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )
