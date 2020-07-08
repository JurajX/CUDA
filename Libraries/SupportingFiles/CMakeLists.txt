# ====================== Supporting Files
set( SupportingFiles ON CACHE BOOL "Flag that SupportingFiles Lib is included" )


add_library( lib_supporting_files_cuda STATIC
    ./include/checkCudaErrors.hpp
    ./include/cudaMemory.hpp
    ./include/timerGPU.hpp
    ./src/cudaMemory.cu
    ./src/timerGPU.cu
)

set_target_properties( lib_supporting_files_cuda PROPERTIES CUDA_SEPERABLE_COMPILATION ON )

target_include_directories( lib_supporting_files_cuda PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )



add_library( lib_supporting_files STATIC
    ./include/functions.hpp
    ./include/timerCPU.hpp
    ./src/functions.cpp
    ./src/timerCPU.cpp
)

target_include_directories( lib_supporting_files PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" )
