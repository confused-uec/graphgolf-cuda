#pragma once

#include "part.hpp"

namespace graphgolf{
    class cudaASPLbeamer{
    private:
        int device;
        uint *d_visited_bits, *d_updated_bits;
        int *d_edges, *h_edges;
        int64_t *d_ret, *h_ret;
    public:
        int N,M,degree;
        cudaASPLbeamer(int N, int M, int degree, int device);
        ~cudaASPLbeamer();
        double calc(part &p);
    };
}
