#pragma once
#include "../cppcode/graph.hpp"

namespace graphgolf{
    class cudaASPLgraph{
        private:
        uint *d_bits, *h_bits;
        uint *d_diff_bits;
        int *d_sum;
        int *d_edges, *h_edges;
        int64_t *h_ret, *d_ret;
        uint *h_ret_bits, *d_ret_bits;
        int N, degree_max, device;
        int nBlock;
    public:
        cudaASPLgraph(int N, int degree_max, int device);
        ~cudaASPLgraph();
        std::pair<int,int64_t> calc(graph &g);
    };
}