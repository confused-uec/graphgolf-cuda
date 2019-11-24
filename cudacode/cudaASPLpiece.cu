#include "cuda_runtime.h"
#include <iostream>
#include "cudaASPLpiece.hpp"
#include "piece.hpp"
#include <bitset>
#include <cassert>
namespace graphgolf{

    __global__ void kernel_aspl_piece_init(uint *bits, int2 N, int width, int2 M, int* sum){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        int Ns=N.x*N.y;
        int Ms=M.x*M.y;
        if(id>=Ns*width) return;
        int v=id>>(__ffs(width)-1);
        int b=id&(width-1);
        int tmp;
        if(Ms>=(b+1)<<5){
            tmp=0;
        }else if(b<<5>=Ms){
            tmp=0xFFFFFFFF;
        }else{
            tmp=~((1<<(Ms-(b<<5)))-1);
        }
        int x=v/N.y,y=v%N.y;
        if(x<M.x&&y<M.y){
            int mv=x*M.y+y;
            if((b<<5)<=mv&&mv<((b+1)<<5)){
                tmp|=1<<(mv-(b<<5));
            }
        }
        sum[id]=0;
        bits[id]=tmp;
    }

    __global__ void kernel_aspl_piece_conv(uint *bits, uint *diff_bits, int *edges, int2 N, int width, int2 M, int degree){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=N.x*N.y*width) return;
        int v=id>>(__ffs(width)-1);
        int b=id&(width-1);
        uint tmp=0;
        int x=v/N.y, y=v%N.y;
        int xm=x%M.x, ym=y%M.y;
        for(int i=0;i<degree;i++){
            //int to = (N+v+edges[vm*degree+i])%N;
            int diff_x = edges[((xm*M.y+ym)*degree+i)*2];
            int diff_y = edges[((xm*M.y+ym)*degree+i)*2+1];
            int to_x = (N.x+x+diff_x)%N.x;
            int to_y = (N.y+y+diff_y)%N.y;
            int to_v = to_x*N.y+to_y;
            tmp|=bits[(to_v<<(__ffs(width)-1))+b];
        }
        tmp&=~bits[id];
        diff_bits[id]=tmp;
    }

    __global__ void kernel_aspl_piece_apply(uint *bits, uint *diff_bits, int *sum, int N, int width, int step){
        int id = blockIdx.x*blockDim.x+threadIdx.x;
        if(id>=N*width) return;
        uint tmp=diff_bits[id];
        bits[id]|=tmp;
        sum[id]+=__popc(tmp)*step;
    }

    //1024 -> 1
    __global__ void kernel_aspl_piece_reduce_plus(int *sum, int64_t *ret, int length){
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

    __global__ void kernel_aspl_piece_reduce_OR(uint *bits, uint *ret, int length){
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

    __global__ void kernel_aspl_piece_reduce_AND(uint *bits, uint *ret, int length){
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

    cudaASPLpiece::cudaASPLpiece(int Nx, int Ny, int Mx, int My, int degree, int device=0):Nx(Nx),Ny(Ny),Mx(Mx),My(My),degree(degree),device(device){
        int N=Nx*Ny;
        int M=Mx*My;
        for(width=1;(width<<5)<M;width<<=1);
        nBlock=(N*width+1023)/1024;
        std::cout<<"width: "<<width<<" nBlock: "<<nBlock<<std::endl;
        cudaSetDevice(device);
        cudaMalloc((void**)&d_bits,N*width*sizeof(uint));
        cudaMalloc((void**)&d_diff_bits,N*width*sizeof(uint));
        cudaMalloc((void**)&d_sum,N*width*sizeof(int));
        cudaMallocHost((void**)&h_bits,N*width*sizeof(uint));
        cudaMalloc((void**)&d_edges,M*degree*sizeof(int)*2);
        cudaMallocHost((void**)&h_edges,M*degree*sizeof(int)*2);
        cudaMalloc((void**)&d_ret,nBlock*sizeof(int64_t));
        cudaMallocHost((void**)&h_ret,nBlock*sizeof(int64_t));
        cudaMalloc((void**)&d_ret_bits,nBlock*sizeof(uint));
        cudaMallocHost((void**)&h_ret_bits,nBlock*sizeof(uint));
    }
    cudaASPLpiece::~cudaASPLpiece(){
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
    std::pair<int,double> cudaASPLpiece::diameterASPL(piece &p){
        cudaSetDevice(device);
        int N=Nx*Ny;
        int M=Mx*My;
        for(int x=0;x<Mx;x++){
            for(int y=0;y<My;y++){
                for(int e=0;e<degree;e++){
                    int dx=0,dy=0;
                    if(e<p.edges[x][y].size()){
                        std::tie(dx,dy)=p.edges[x][y][e];
                    }
                    h_edges[((x*My+y)*degree+e)*2]=dx;
                    h_edges[((x*My+y)*degree+e)*2+1]=dy;
                }
            }
        }
        cudaMemcpy(d_edges,h_edges,M*degree*sizeof(int)*2,cudaMemcpyHostToDevice);
        //15625x256 = 4x1,000,000
        kernel_aspl_piece_init<<<nBlock,1024>>>(d_bits,make_int2(Nx,Ny),width,make_int2(Mx,My),d_sum);
        //std::cout<<"N: "<<N<<" M: "<<M<<" width: "<<width<<" nBlock: "<<nBlock<<std::endl;
        cudaDeviceSynchronize();
        int diameter=100000000;
        for(int step=1;step<100;step++){
            kernel_aspl_piece_conv<<<nBlock,1024>>>(d_bits,d_diff_bits,d_edges,make_int2(Nx,Ny),width,make_int2(Mx,My),degree);
            cudaDeviceSynchronize();
            kernel_aspl_piece_apply<<<nBlock,1024>>>(d_bits,d_diff_bits,d_sum,N,width,step);
            cudaDeviceSynchronize();
            kernel_aspl_piece_reduce_OR<<<nBlock,1024>>>(d_diff_bits,d_ret_bits,N*width);
            cudaDeviceSynchronize();
            cudaMemcpy(h_ret_bits,d_ret_bits,nBlock*sizeof(uint),cudaMemcpyDeviceToHost);
            uint flag = 0;
            for(int i=0;i<nBlock;i++) flag|=h_ret_bits[i];
            if(flag==0){
                kernel_aspl_piece_reduce_AND<<<nBlock,1024>>>(d_bits,d_ret_bits,N*width);
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
        kernel_aspl_piece_reduce_plus<<<nBlock,1024>>>(d_sum,d_ret,N*width);
        cudaMemcpy(h_ret,d_ret,nBlock*sizeof(int64_t),cudaMemcpyDeviceToHost);
        int64_t total=0;
        for(int i=0;i<nBlock;i++) total+=h_ret[i];
        total*=N/M;
        return std::make_pair(diameter,double(total)/(int64_t(N)*(N-1)));
    }
}