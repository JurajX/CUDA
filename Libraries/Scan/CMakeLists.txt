# ====================== Scan
set( Scan ON CACHE BOOL "Flag that Scan Lib is included" )


add_library( lib_scan STATIC
    ./include/scan.hpp
    ./src/scan_cuda.cu
    ./src/scan_std.cpp
)

if( NOT SupportingFiles )
    add_subdirectory(
        "${CMAKE_CURRENT_SOURCE_DIR}/../SupportingFiles"
        "${CMAKE_CURRENT_SOURCE_DIR}/../SupportingFiles/build")
endif( NOT SupportingFiles )













set_target_properties( lib_scan PROPERTIES CUDA_SEPERABLE_COMPILATION ON )


find_package( PkgConfig REQUIRED )                                  # the `pkg_check_modules` function is created with this call
pkg_check_modules( tbb REQUIRED IMPORTED_TARGET tbb )               # these calls create special `PkgConfig::<MODULE>` variables
target_link_libraries( lib_scan INTERFACE PkgConfig::tbb )


target_link_libraries( lib_scan PRIVATE lib_supporting_files_cuda )
target_link_libraries( lib_scan PRIVATE lib_supporting_files )



target_include_directories( lib_scan PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )
