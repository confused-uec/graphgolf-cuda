#pragma once
#include <iostream>
#include <tuple>
#include "../cppcode/piece.hpp"
namespace graphgolf{
    class cudaASPLpiece{
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
        int Nx,Ny,Mx,My,degree;
        cudaASPLpiece(int Nx, int Ny, int Mx, int My, int degree, int device);
        ~cudaASPLpiece();
        std::pair<int,double> diameterASPL(piece &p);
    };
}