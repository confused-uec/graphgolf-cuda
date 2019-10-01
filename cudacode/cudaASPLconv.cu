#include "cuda_runtime.h"
#include <iostream>
#include "cudaASPLconv.hpp"
#include "part.hpp"
#include <bitset>
namespace graphgolf{

    __global__ void kernel_aspl_init(uint *bits, int N, int width, int M, int* sum){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=N*width) return;
        int v=id>>(__ffs(width)-1);
        int b=id&(width-1);
        int tmp;
        if(M>=(b+1)<<5){
            tmp=0;
        }else if(b<<5>=M){
            tmp=0xFFFFFFFF;
        }else{
            tmp=~((1<<(M-(b<<5)))-1);
        }
        /*
        if(((b+1)<<5)>M){
            tmp=~((1<<(M-(b<<5)))-1);
        }else{
            tmp=0;
        }*/
        if((b<<5)<=v&&v<((b+1)<<5)){
            tmp|=1<<(v-(b<<5));
        }
        sum[id]=0;
        bits[id]=tmp;
    }

    __global__ void kernel_aspl_conv(uint *bits, uint *diff_bits, int *edges, int N, int width, int M, int degree){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=N*width) return;
        int v=id>>(__ffs(width)-1);
        int b=id&(width-1);
        uint tmp=0;
        int vm=v%M;
        for(int i=0;i<degree;i++){
            int to = (N+v+edges[vm*degree+i])%N;
            tmp|=bits[(to<<(__ffs(width)-1))+b];
        }
        tmp&=~bits[id];
        diff_bits[id]=tmp;
    }

    __global__ void kernel_aspl_apply(uint *bits, uint *diff_bits, int *sum, int N, int width, int step){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=N*width) return;
        uint tmp=diff_bits[id];
        bits[id]|=tmp;
        sum[id]+=__popc(tmp)*step;
    }

    //1024 -> 1
    __global__ void kernel_aspl_reduce_plus(int *sum, int64_t *ret, int length){
        __shared__ int64_t tmp[32];
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        int64_t elem = id<length?sum[id]:0;
        #pragma unroll
        for(int delta=16;delta;delta>>=1){
            elem+=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        int laneid=threadIdx.x&0x1F;
        int warpid=threadIdx.x>>5;
        if(laneid==0) tmp[warpid]=elem;
        __syncthreads();
        if(warpid) return;
        elem=tmp[laneid];
        #pragma unroll
        for(int delta=16;delta;delta>>=1){
            elem+=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        if(threadIdx.x==0) ret[blockIdx.x]=elem;
    }

    __global__ void kernel_aspl_reduce_OR(uint *bits, uint *ret, int length){
        __shared__ uint tmp[32];
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        uint elem = id<length?bits[id]:0;
        #pragma unroll
        for(int delta=16;delta;delta>>=1){
            elem|=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        int laneid=threadIdx.x&0x1F;
        int warpid=threadIdx.x>>5;
        if(laneid==0) tmp[warpid]=elem;
        __syncthreads();
        if(warpid) return;
        elem=tmp[laneid];
        #pragma unroll
        for(int delta=16;delta;delta>>=1){
            elem|=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        if(threadIdx.x==0) ret[blockIdx.x]=elem;
    }

    __global__ void kernel_aspl_reduce_AND(uint *bits, uint *ret, int length){
        __shared__ uint tmp[32];
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        uint elem = id<length?bits[id]:0xFFFFFFFF;
        #pragma unroll
        for(int delta=16;delta;delta>>=1){
            elem&=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        int laneid=threadIdx.x&0x1F;
        int warpid=threadIdx.x>>5;
        if(laneid==0) tmp[warpid]=elem;
        __syncthreads();
        if(warpid) return;
        elem=tmp[laneid];
        #pragma unroll
        for(int delta=16;delta;delta>>=1){
            elem&=__shfl_down_sync(0xFFFFFFFF, elem, delta);
        }
        if(threadIdx.x==0) ret[blockIdx.x]=elem;
    }

    cudaASPLconv::cudaASPLconv(int N, int M, int degree, int device=0):N(N),M(M),degree(degree),device(device){
        for(width=1;(width<<5)<M;width<<=1);
        nBlock=(N*width+1023)/1024;
        std::cout<<"width: "<<width<<" nBlock: "<<nBlock<<std::endl;
        cudaSetDevice(device);
        cudaMalloc((void**)&d_bits,N*width*sizeof(uint));
        cudaMalloc((void**)&d_diff_bits,N*width*sizeof(uint));
        cudaMalloc((void**)&d_sum,N*width*sizeof(int));
        cudaMallocHost((void**)&h_bits,N*width*sizeof(uint));
        cudaMalloc((void**)&d_edges,M*degree*sizeof(int));
        cudaMallocHost((void**)&h_edges,M*degree*sizeof(int));
        cudaMalloc((void**)&d_ret,nBlock*sizeof(int64_t));
        cudaMallocHost((void**)&h_ret,nBlock*sizeof(int64_t));
        cudaMalloc((void**)&d_ret_bits,nBlock*sizeof(uint));
        cudaMallocHost((void**)&h_ret_bits,nBlock*sizeof(uint));
    }
    cudaASPLconv::~cudaASPLconv(){
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
    double cudaASPLconv::calc(part &p){
        cudaSetDevice(device);
        for(int i=0;i<M;i++){
            for(int j=0;j<degree;j++) h_edges[i*degree+j]=p.edges[i][j];
        }
        cudaMemcpy(d_edges,h_edges,M*degree*sizeof(int),cudaMemcpyHostToDevice);
        //15625x256 = 4x1,000,000
        kernel_aspl_init<<<nBlock,1024>>>(d_bits,N,width,M,d_sum);
        //std::cout<<"N: "<<N<<" M: "<<M<<" width: "<<width<<" nBlock: "<<nBlock<<std::endl;
        cudaDeviceSynchronize();
        for(int step=1;step<100;step++){
            kernel_aspl_conv<<<nBlock,1024>>>(d_bits,d_diff_bits,d_edges,N,width,M,degree);
            cudaDeviceSynchronize();
            kernel_aspl_apply<<<nBlock,1024>>>(d_bits,d_diff_bits,d_sum,N,width,step);
            cudaDeviceSynchronize();
            kernel_aspl_reduce_OR<<<nBlock,1024>>>(d_diff_bits,d_ret_bits,N*width);
            cudaDeviceSynchronize();
            cudaMemcpy(h_ret_bits,d_ret_bits,nBlock*sizeof(uint),cudaMemcpyDeviceToHost);
            uint flag = 0;
            for(int i=0;i<nBlock;i++) flag|=h_ret_bits[i];
            if(flag==0){
                kernel_aspl_reduce_AND<<<nBlock,1024>>>(d_bits,d_ret_bits,N*width);
                cudaDeviceSynchronize();
                cudaMemcpy(h_ret_bits,d_ret_bits,nBlock*sizeof(uint),cudaMemcpyDeviceToHost);
                flag=0xFFFFFFFF;
                for(int i=0;i<nBlock;i++) flag&=h_ret_bits[i];
                if(flag==0xFFFFFFFF){
                    //std::cout<<"Diameter: "<<step-1<<std::endl;
                    ;
                }else{
                    std::cout<<"Graph is unconnected!"<<std::endl;
                    return 1e9;
                }
                break;
            }else if(step==200){
                std::cout<<"Too Large Diameter!"<<std::endl;
                return 1e9;
            }
        }
        kernel_aspl_reduce_plus<<<nBlock,1024>>>(d_sum,d_ret,N*width);
        cudaMemcpy(h_ret,d_ret,nBlock*sizeof(int64_t),cudaMemcpyDeviceToHost);
        int64_t total=0;
        for(int i=0;i<nBlock;i++) total+=h_ret[i];
        total*=N/M;
        return double(total)/(int64_t(N)*(N-1));
    }
    std::pair<int,double> cudaASPLconv::diameterASPL(part &p){
        cudaSetDevice(device);
        for(int i=0;i<M;i++){
            for(int j=0;j<degree;j++) h_edges[i*degree+j]=p.edges[i][j];
        }
        cudaMemcpy(d_edges,h_edges,M*degree*sizeof(int),cudaMemcpyHostToDevice);
        //15625x256 = 4x1,000,000
        kernel_aspl_init<<<nBlock,1024>>>(d_bits,N,width,M,d_sum);
        //std::cout<<"N: "<<N<<" M: "<<M<<" width: "<<width<<" nBlock: "<<nBlock<<std::endl;
        cudaDeviceSynchronize();
        int diameter=100000000;
        for(int step=1;step<100;step++){
            kernel_aspl_conv<<<nBlock,1024>>>(d_bits,d_diff_bits,d_edges,N,width,M,degree);
            cudaDeviceSynchronize();
            kernel_aspl_apply<<<nBlock,1024>>>(d_bits,d_diff_bits,d_sum,N,width,step);
            cudaDeviceSynchronize();
            kernel_aspl_reduce_OR<<<nBlock,1024>>>(d_diff_bits,d_ret_bits,N*width);
            cudaDeviceSynchronize();
            cudaMemcpy(h_ret_bits,d_ret_bits,nBlock*sizeof(uint),cudaMemcpyDeviceToHost);
            uint flag = 0;
            for(int i=0;i<nBlock;i++) flag|=h_ret_bits[i];
            if(flag==0){
                kernel_aspl_reduce_AND<<<nBlock,1024>>>(d_bits,d_ret_bits,N*width);
                cudaDeviceSynchronize();
                cudaMemcpy(h_ret_bits,d_ret_bits,nBlock*sizeof(uint),cudaMemcpyDeviceToHost);
                flag=0xFFFFFFFF;
                for(int i=0;i<nBlock;i++) flag&=h_ret_bits[i];
                if(flag==0xFFFFFFFF){
                    //std::cout<<"Diameter: "<<step-1<<std::endl;
                    diameter=step-1;
                }else{
                    std::cout<<"Graph is unconnected!"<<std::endl;
                    return std::make_pair(diameter,100000000.0);
                }
                break;
            }else if(step==200){
                std::cout<<"Too Large Diameter!"<<std::endl;
                return std::make_pair(diameter,100000000.0);
            }
        }
        kernel_aspl_reduce_plus<<<nBlock,1024>>>(d_sum,d_ret,N*width);
        cudaMemcpy(h_ret,d_ret,nBlock*sizeof(int64_t),cudaMemcpyDeviceToHost);
        int64_t total=0;
        for(int i=0;i<nBlock;i++) total+=h_ret[i];
        total*=N/M;
        return std::make_pair(diameter,double(total)/(int64_t(N)*(N-1)));
    }
}