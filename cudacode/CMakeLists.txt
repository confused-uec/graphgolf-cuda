find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message("Cuda version: " ${CUDA_VERSION})
    CUDA_ADD_LIBRARY(cudaASPL STATIC
        cudaASPLconv.cu 
        cudaASPLconv.hpp 
        cudaASPLbeamer.cu 
        cudaASPLbeamer.hpp 
        cudaASPLpiece.cu
        cudaASPLpiece.hpp
        cudaASPLbox.cu
        cudaASPLbox.hpp
    )
    set(CUDA_HOST_COMPILER clang++)
    target_compile_options(cudaASPL PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_35,code=sm_35 -O2>)
    include_directories(${CMAKE_SOURCE_DIR}/cppcode)
    cuda_include_directories(${CMAKE_SOURCE_DIR}/cppcode)
    CUDA_ADD_LIBRARY(cudaASPLgraph STATIC cudaASPLgraph.cu cudaASPLgraph.hpp)
    target_compile_options(cudaASPLgraph PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_35,code=sm_35 -O2>)
endif()