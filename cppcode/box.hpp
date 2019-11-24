#pragma once
#include <iostream>
#include <vector>
#include <tuple>
namespace graphgolf{
    struct box{
        public:
        int Nx,Ny,Nz,Mx,My,Mz,N,M;
        bool regular;
        int degree=0;
        std::vector<std::vector<std::vector<std::vector<std::tuple<int,int,int>>>>> edges;
        //edges[x][y][z][e]:=(diffx,diffy,diffz)
        box(){}
        box(int Nx,int Ny,int Nz,int Mx,int My, int Mz):Nx(Nx),Ny(Ny),Nz(Nz),Mx(Mx),My(My),Mz(Mz){
            N=Nx*Ny*Nz; M=Mx*My*Mz;
            edges.resize(Mx);
            for(auto &e:edges){
                e.resize(My);
                for(auto &f:e) f.resize(Mz);
            }
        }
        std::tuple<int,int,int> id2xyz(int id){
            int x=id/(Ny*Nz);
            id-=x*Ny*Nz;
            int y=id/Nz;
            int z=id%Nz;
            return std::make_tuple(x,y,z);
        }
        int xyz2id(int x,int y, int z){
            return x*Ny*Nz+y*Nz+z;
        }
        int xyz2id(std::tuple<int,int,int> t){
            int x,y,z;
            std::tie(x,y,z)=t;
            return xyz2id(x,y,z);            
        }
        void load(std::istream &ist){
            ist>>Nx>>Ny>>Nz>>Mx>>My>>Mz;
            N=Nx*Ny*Nz; M=Mx*My*Mz;
            edges.resize(Mx);
            for(auto &e:edges){
                e.resize(My);
                for(auto &f:e) f.resize(Mz);
            }
            regular=true;
            for(int x=0;x<Mx;x++){
                for(int y=0;y<My;y++){
                    for(int z=0;z<Mz;z++){
                        int E;
                        ist>>E;
                        if(!degree)degree=E;
                        else if(degree!=E){
                            regular=false;
                        }
                        for(int e=0;e<E;e++){
                            int diffx,diffy,diffz;
                            ist>>diffx>>diffy>>diffz;
                            edges[x][y][z].emplace_back(diffx,diffy,diffz);
                        }
                    }
                }
            }
            if(!regular)std::cout<<"non-regular graph"<<std::endl;
        }
        void print(std::ostream &ost){
            ost<<Nx<<' '<<Ny<<' '<<Nz<<' '<<Mx<<' '<<My<<' '<<Mz<<std::endl;
            for(int x=0;x<Mx;x++){
                for(int y=0;y<My;y++){
                    for(int z=0;z<Mz;z++){
                        ost<<edges[x][y][z].size();
                        for(auto t:edges[x][y][z]){
                            int diffx,diffy,diffz;
                            std::tie(diffx,diffy,diffz)=t;
                            ost<<' '<<diffx<<' '<<diffy<<' '<<diffz;
                        }
                        ost<<std::endl;
                    }
                }
            }
        }
    };
}