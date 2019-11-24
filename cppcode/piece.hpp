#pragma once
#include <iostream>
#include <vector>

namespace graphgolf{
    struct piece{
        public:
        int Nx,Ny,Mx,My;
        bool regular;
        int degree;
        std::vector<std::vector<std::vector<std::pair<int,int>>>> edges;
        //edges[x][y][e]:=(diffx,diffy)
        piece(){}
        piece(int Nx,int Ny,int Mx,int My):Nx(Nx),Ny(Ny),Mx(Mx),My(My){
            edges.resize(Mx);
            for(auto &e:edges) e.resize(My);
        }
        void load(std::istream &ist){
            ist>>Nx>>Ny>>Mx>>My;
            edges.resize(Mx);
            for(auto &e:edges) e.resize(My);
            degree=0;
            regular=true;
            for(int x=0;x<Mx;x++){
                for(int y=0;y<My;y++){
                    int E;
                    ist>>E;
                    if(!degree)degree=E;
                    else if(degree!=E){
                        regular=false;
                    }
                    for(int e=0;e<E;e++){
                        int diffx,diffy;
                        ist>>diffx>>diffy;
                        edges[x][y].emplace_back(diffx,diffy);
                    }
                }
            }
            if(!regular)std::cout<<"non-regular graph"<<std::endl;
        }
        void print(std::ostream &ost){
            ost<<Nx<<' '<<Ny<<' '<<Mx<<' '<<My<<std::endl;
            for(int x=0;x<Mx;x++){
                for(int y=0;y<My;y++){
                    ost<<edges[x][y].size();
                    for(auto diff:edges[x][y]){
                        ost<<' '<<diff.first<<' '<<diff.second;
                    }
                    ost<<std::endl;
                }
            }
        }
    };
}