#pragma once
#include <iostream>
#include <tuple>
#include "../cppcode/box.hpp"
namespace graphgolf{
    class cudaASPLbox{
    private:
        uint *d_bits, *h_bits;
        uint *d_diff_bits;
        int *d_sum;
        int *d_edges, *h_edges;
        int64_t *h_ret, *d_ret;
        uint *h_ret_bits, *d_ret_bits;
        int device;
        int width,nBlock;
    public:
        int Nx,Ny,Nz,Mx,My,Mz,degree;
        cudaASPLbox(int Nx, int Ny, int Nz, int Mx, int My, int Mz, int degree, int device);
        ~cudaASPLbox();
        std::pair<int,double> diameterASPL(box &p);
    };
}