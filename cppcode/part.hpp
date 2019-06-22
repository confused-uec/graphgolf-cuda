#pragma once
#include <iostream>
#include <vector>

namespace graphgolf{
    struct part{
        public:
        int N,M;
        bool regular;
        int degree;
        std::vector<std::vector<int>> edges;
        part(){}
        void load(std::istream &ist){
            ist>>N>>M;
            edges.resize(M);
            degree=0;
            regular=true;
            for(int i=0;i<M;i++){
                int E;
                ist>>E;
                if(!degree)degree=E;
                else if(degree!=E){
                    regular=false;
                    std::cout<<"non-regular graph"<<std::endl;
                }
                for(int j=0;j<E;j++){
                    int diff;
                    ist>>diff;
                    edges[i].push_back(diff);
                }
            }
        }
        void print(std::ostream &ost){
            ost<<N<<' '<<M<<std::endl;
            for(int i=0;i<M;i++){
                ost<<edges[i].size();
                for(auto diff:edges[i]){
                    ost<<' '<<diff;
                }
                ost<<std::endl;
            }
        }
    };
}