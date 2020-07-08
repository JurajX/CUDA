# ====================== HW
set( HW ON CACHE BOOL "Flag that HW Lib is included" )


add_library( lib_hw STATIC
    ./include/blur.hpp
    ./include/filter.hpp
    ./include/grayscale.hpp
    ./include/hdr.hpp
    ./include/HW.hpp
    ./include/image.hpp
    ./include/red_eye_removal.hpp
    ./src/blur.cu
    ./src/filter.cpp
    ./src/grayscale.cu
    ./src/hdr.cu
    ./src/HW.cpp
    ./src/image.cpp
    ./src/red_eye_removal.cu
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

if( NOT Histogram )
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../Histogram"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Histogram/build")
endif( NOT Histogram )

if( NOT Sort )
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../Sort"
        "${CMAKE_CURRENT_SOURCE_DIR}/../Sort/build")
endif( NOT Sort )

set_target_properties( lib_hw PROPERTIES CUDA_SEPERABLE_COMPILATION ON )
set_source_files_properties(./src/red_eye_removal.cu PROPERTIES COMPILE_FLAGS "--expt-extended-lambda")

find_package( PkgConfig REQUIRED )                                  # the `pkg_check_modules` function is created with this call
pkg_check_modules( opencv4 REQUIRED IMPORTED_TARGET opencv4 )       # these calls create special `PkgConfig::<MODULE>` variables
target_link_libraries( lib_hw PRIVATE PkgConfig::opencv4 )

target_link_libraries( lib_hw PRIVATE lib_supporting_files_cuda )
target_link_libraries( lib_hw PRIVATE lib_supporting_files )
target_link_libraries( lib_hw PRIVATE lib_reduce )
target_link_libraries( lib_hw PRIVATE lib_scan )
target_link_libraries( lib_hw PRIVATE lib_histogram )
target_link_libraries( lib_hw PRIVATE lib_sort )

target_include_directories( lib_hw PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )
