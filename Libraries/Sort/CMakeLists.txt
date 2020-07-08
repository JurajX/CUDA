# ====================== Sort
set( Sort ON CACHE BOOL "Flag that Sort Lib is included" )


add_library( lib_sort STATIC
    ./include/sort.hpp
    ./src/sort_cuda.cu
    ./src/sort_std.cpp
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

if( NOT Scan )
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../Scan"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Scan/build")
endif( NOT Scan )

set_target_properties( lib_sort PROPERTIES CUDA_SEPERABLE_COMPILATION ON )


find_package( PkgConfig REQUIRED )                                  # the `pkg_check_modules` function is created with this call
pkg_check_modules( tbb REQUIRED IMPORTED_TARGET tbb )               # these calls create special `PkgConfig::<MODULE>` variables
target_link_libraries( lib_sort INTERFACE PkgConfig::tbb )


target_link_libraries( lib_sort PRIVATE lib_supporting_files_cuda )
target_link_libraries( lib_sort PRIVATE lib_supporting_files )
target_link_libraries( lib_sort PRIVATE lib_reduce )
target_link_libraries( lib_sort PRIVATE lib_scan )

target_include_directories( lib_sort PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )
