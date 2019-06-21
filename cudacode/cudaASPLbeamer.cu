#include "cuda_runtime.h"
#include "part.hpp"
#include "cudaASPLbeamer.hpp"
#include <iostream>

namespace graphgolf{
    __global__ void kernel_aspl(uint *visited, uint *update, int *edges, int N, int M, int degree, int64_t *ret){
        __shared__ int64_t sum[1024];
        int m = blockIdx.x;
        visited+=blockIdx.x*31250;
        update+=blockIdx.x*31250;
        for(int idx=threadIdx.x;idx<31250;idx+=blockDim.x){
            visited[idx]=(idx==(m>>5))?(1<<(m&0x1F)):0;
        }
        __syncthreads();
        int sum_tmp=0;
        for(int steps=1;steps<=7;steps++){
            for(int idx=threadIdx.x;idx<31250;idx+=blockDim.x){
                uint update_tmp=0;
                uint visited_tmp=visited[idx];
                for(int i=0;i<32;i++){
                    if((visited_tmp>>i)&1) continue;
                    int v = (idx<<5)+i;
                    int vmod = v%M;
                    for(int d=0;d<degree;d++){
                        int diff = edges[vmod*degree+d];
                        int to = (v+diff+N)%N;
                        if(visited[to>>5]&(1<<(to&0x1F))){
                            update_tmp|=(1<<i);
                        }
                    }
                }
                //update[idx]=update_tmp;
                update[idx]=update_tmp;
            }
            __syncthreads();
            for(int idx=threadIdx.x;idx<31250;idx+=blockDim.x){
                sum_tmp+=__popc(update[idx])*steps;
                visited[idx]|=update[idx];
            }
            __syncthreads();
        }
        sum[threadIdx.x]=sum_tmp;
        __syncthreads();
        int r=blockDim.x/2;
        while(r){
            if(threadIdx.x<r) sum[threadIdx.x]+=sum[threadIdx.x+r];
            __syncthreads();
            r>>=1;
        }
        __syncthreads();
        if(threadIdx.x==0) ret[blockIdx.x]=sum[0];
    }

    cudaASPLbeamer::cudaASPLbeamer(int N, int M, int degree, int device=0): N(N),M(M),degree(degree),device(device){
        cudaMalloc((void**)&d_visited_bits,M*31250*sizeof(uint));
        cudaMalloc((void**)&d_updated_bits,M*31250*sizeof(uint));
        cudaMalloc((void**)&d_edges,M*degree*sizeof(int));
        cudaMallocHost((void**)&h_edges,M*degree*sizeof(int));
        cudaMalloc((void**)&d_ret,M*sizeof(int64_t));
        cudaMallocHost((void**)&h_ret,M*sizeof(int64_t));
    }
    cudaASPLbeamer::~cudaASPLbeamer(){
        cudaFree(d_visited_bits);
        cudaFree(d_updated_bits);
        cudaFree(d_edges);
        cudaFreeHost(h_edges);
        cudaFree(d_ret);
        cudaFreeHost(h_ret);
    }
    double cudaASPLbeamer::calc(part &p){
        if(!p.regular) return 100000000;
        for(int i=0;i<M;i++){
            for(int j=0;j<degree;j++) h_edges[i*degree+j]=p.edges[i][j];
        }
        cudaMemcpy(d_edges,h_edges,M*degree*sizeof(int),cudaMemcpyHostToDevice);
        kernel_aspl<<<M,1024>>>(d_visited_bits,d_updated_bits,d_edges,N,M,degree,d_ret);
        cudaMemcpy(h_ret,d_ret,M*sizeof(int64_t),cudaMemcpyDeviceToHost);

        int64_t total = 0;
        for(int i=0;i<M;i++) total+=h_ret[i];
        total*=N/M;
        return double(total)/(int64_t(N)*(N-1));
    }
}