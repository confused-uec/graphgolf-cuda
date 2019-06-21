#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <tuple>
#include <algorithm>

namespace graphgolf{

    struct graph{
        std::vector<std::vector<int>> edges;
        int N;
        int degree_max;
        void load(std::istream &ist){
            std::string s;
            std::vector<std::tuple<int,int>> ve;
            N=0;
            while(std::getline(ist,s)){
                // std::cout<<s<<std::endl;
                std::stringstream ss;
                ss.str(s);
                int u,v;
                ss>>u>>v;
                // std::cout<<u<<' '<<v<<std::endl;
                ve.emplace_back(u,v);
                N=std::max({N,u,v});
            }
            // std::cout<<"N: "<<N<<std::endl;
            N++;
            edges.resize(N);
            for(auto t:ve){
                int u,v;
                std::tie(u,v)=t;
                edges[u].push_back(v);
                edges[v].push_back(u);
            }
            degree_max=0;
            for(auto &e:edges) degree_max=std::max(degree_max,int(e.size()));
        }
        bool is_regular(){
            for(auto &e:edges) if(degree_max!=e.size()) return false;
            return true;
        }
        void print(std::ostream &ost){
            for(int v=0;v<edges.size();v++){
                for(auto u:edges[v]){
                    if(u<v) ost<<u<<' '<<v<<std::endl;
                }
            }
        }
    };
}

