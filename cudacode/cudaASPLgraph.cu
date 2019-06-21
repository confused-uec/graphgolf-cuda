#include "cudaASPLgraph.hpp"
#include "cuda_runtime.h"
#include <bitset>
namespace graphgolf{
    ///<<<N,32>>で呼び出す?
    __global__ void kernel_aspl_init_bits(uint *bits, int N, int offset){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=N<<5) return;
        int v=(id>>5)-offset;
        int b=id&0x1F;
        int tmp;
        int M=(N-offset<1024?N-offset:1024);
        if(M>=(b+1)<<5){
            tmp=0;
        }else if(b<<5>=M){
            tmp=0xFFFFFFFF;
        }else{
            tmp=~((1<<(M-(b<<5)))-1);
        }
        if((b<<5)<=v&&v<((b+1)<<5)){
            tmp|=1<<(v-(b<<5));
        }
        // sum[id]=0;
        bits[id]=tmp;
    }

    __global__ void kernel_aspl_conv(uint *bits, uint *diff_bits, int *edges, int N, int degree){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=N<<5) return;
        int v=id>>5;
        int b=id&0x1F;
        uint tmp=0;
        for(int i=0;i<degree;i++){
            int to = edges[v*degree+i];
            tmp|=bits[(to<<5)+b];
        }
        tmp&=~bits[id];
        diff_bits[id]=tmp;
    }

    __global__ void kernel_aspl_apply(uint *bits, uint *diff_bits, int *sum, int N, int step){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=N<<5) return;
        uint tmp=diff_bits[id];
        bits[id]|=tmp;
        sum[id]+=__popc(tmp)*step;
    }

    //1024 -> 1
    __global__ void kernel_aspl_reduce_plus(int *sum, int64_t *ret, int length){
        __shared__ int64_t tmp[32];
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        int64_t elem = id<length?sum[id]:0;
        #pragma unroll 16
        for(int delta=16;delta;delta>>=1){
            elem+=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        int laneid=threadIdx.x&0x1F;
        int warpid=threadIdx.x>>5;
        if(laneid==0) tmp[warpid]=elem;
        __syncthreads();
        if(warpid) return;
        elem=tmp[laneid];
        #pragma unroll 16
        for(int delta=16;delta;delta>>=1){
            elem+=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        if(threadIdx.x==0) ret[blockIdx.x]=elem;
    }

    __global__ void kernel_aspl_reduce_OR(uint *bits, uint *ret, int length){
        __shared__ uint tmp[32];
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        uint elem = id<length?bits[id]:0;
        #pragma unroll 16
        for(int delta=16;delta;delta>>=1){
            elem|=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        int laneid=threadIdx.x&0x1F;
        int warpid=threadIdx.x>>5;
        if(laneid==0) tmp[warpid]=elem;
        __syncthreads();
        if(warpid) return;
        elem=tmp[laneid];
        #pragma unroll 16
        for(int delta=16;delta;delta>>=1){
            elem|=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        if(threadIdx.x==0) ret[blockIdx.x]=elem;
    }

    __global__ void kernel_aspl_reduce_AND(uint *bits, uint *ret, int length){
        __shared__ uint tmp[32];
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        uint elem = id<length?bits[id]:0xFFFFFFFF;
        #pragma unroll 16
        for(int delta=16;delta;delta>>=1){
            elem&=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        int laneid=threadIdx.x&0x1F;
        int warpid=threadIdx.x>>5;
        if(laneid==0) tmp[warpid]=elem;
        __syncthreads();
        if(warpid) return;
        elem=tmp[laneid];
        #pragma unroll 16
        for(int delta=16;delta;delta>>=1){
            elem&=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        if(threadIdx.x==0) ret[blockIdx.x]=elem;
    }


    cudaASPLgraph::cudaASPLgraph(int N, int degree_max, int device): N(N), degree_max(degree_max), device(device){
        nBlock = (N+31)/32;//各ブロックに32頂点を押し込む
        cudaSetDevice(device);
        cudaMalloc((void**)&d_bits,N*32*sizeof(uint));
        cudaMalloc((void**)&d_diff_bits,N*32*sizeof(uint));
        cudaMalloc((void**)&d_sum,N*32*sizeof(int));
        cudaMallocHost((void**)&h_bits,N*32*sizeof(uint));
        cudaMalloc((void**)&d_edges,N*degree_max*sizeof(int));
        cudaMallocHost((void**)&h_edges,N*degree_max*sizeof(int));
        cudaMalloc((void**)&d_ret,nBlock*sizeof(int64_t));
        cudaMallocHost((void**)&h_ret,nBlock*sizeof(int64_t));
        cudaMalloc((void**)&d_ret_bits,nBlock*sizeof(uint));
        cudaMallocHost((void**)&h_ret_bits,nBlock*sizeof(uint));
    }
    cudaASPLgraph::~cudaASPLgraph(){
        cudaSetDevice(device);
        cudaFree(d_bits);
        cudaFree(d_diff_bits);
        cudaFree(d_sum);
        cudaFreeHost(h_bits);
        cudaFree(d_edges);
        cudaFreeHost(h_edges);
        cudaFree(d_ret);
        cudaFreeHost(h_ret);
        cudaFree(d_ret_bits);
        cudaFreeHost(h_ret_bits);
    }
    std::pair<int,int64_t> cudaASPLgraph::calc(graph &g){//直径, totalを返す
        //std::cout<<"N: "<<N<<" degree_max: "<<degree_max<<" nBlock: "<<nBlock<<std::endl;
        cudaSetDevice(device);
        for(int i=0;i<N;i++){
            for(int j=0;j<degree_max;j++){
                if(j<g.edges[i].size()){
                    h_edges[i*degree_max+j]=g.edges[i][j];
                }else{
                    //余った場所は自己ループ辺で埋める(あまり良くない)
                    h_edges[i*degree_max+j]=i;
                }
            }
        }
        cudaMemcpy(d_edges,h_edges,N*degree_max*sizeof(int),cudaMemcpyHostToDevice);
        int diameter=0;
        int64_t total=0;
        cudaMemset(d_sum,0,N*32*sizeof(int));
        for(int offset=0;offset<N;offset+=1024){
            if(offset)std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
            std::cout<<offset<<'/'<<N<<std::flush;
            kernel_aspl_init_bits<<<nBlock,1024>>>(d_bits,N,offset);
            cudaDeviceSynchronize();
            for(int step=1;step<200;step++){
                kernel_aspl_conv<<<nBlock,1024>>>(d_bits,d_diff_bits,d_edges,N,degree_max);
                cudaDeviceSynchronize();
                kernel_aspl_apply<<<nBlock,1024>>>(d_bits,d_diff_bits,d_sum,N,step);
                cudaDeviceSynchronize();
                kernel_aspl_reduce_OR<<<nBlock,1024>>>(d_diff_bits,d_ret_bits,N<<5);
                cudaDeviceSynchronize();
                cudaMemcpy(h_ret_bits,d_ret_bits,nBlock*sizeof(uint),cudaMemcpyDeviceToHost);
                uint flag = 0;
                for(int i=0;i<nBlock;i++) flag|=h_ret_bits[i];
                if(flag==0){
                    kernel_aspl_reduce_AND<<<nBlock,1024>>>(d_bits,d_ret_bits,N<<5);
                    cudaDeviceSynchronize();
                    cudaMemcpy(h_ret_bits,d_ret_bits,nBlock*sizeof(uint),cudaMemcpyDeviceToHost);
                    flag=0xFFFFFFFF;
                    for(int i=0;i<nBlock;i++) flag&=h_ret_bits[i];
                    if(flag!=0xFFFFFFFF){
                        //Graph is unconnected
                        std::cout<<"Graph is unconnected"<<std::endl;
                        return std::make_pair(-1,-1);
                    }else{
                        diameter=std::max(diameter,step-1);
                        break;
                    }
                }else if(step+1==200){
                    //too large diameter
                    std::cout<<"Too large diameter"<<std::endl;
                    return std::make_pair(-2,-2);
                }
            }
        }
        std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
        kernel_aspl_reduce_plus<<<nBlock,1024>>>(d_sum,d_ret,N<<5);
        cudaMemcpy(h_ret,d_ret,nBlock*sizeof(int64_t),cudaMemcpyDeviceToHost);
        for(int i=0;i<nBlock;i++) total+=h_ret[i];    
        return std::make_pair(diameter,total);
    }
}