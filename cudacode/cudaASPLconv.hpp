#pragma once
#include <iostream>
#include "../cppcode/part.hpp"

namespace graphgolf{
    class cudaASPLconv{
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
        int N,M,degree;
        cudaASPLconv(int N, int M, int degree, int device);
        ~cudaASPLconv();
        double calc(part &p);
        std::pair<int,double> diameterASPL(part &p);
    };
}