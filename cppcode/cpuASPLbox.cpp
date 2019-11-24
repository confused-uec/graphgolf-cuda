#include <bitset>
#include <array>
#include <vector>
#include <limits>
#include "arrayQueue.hpp"
#include "box.hpp"
#include <tuple>
#include <cassert>
namespace graphgolf{
    template<int capacity> class cpuASPLbox{
    public:
        std::pair<int,double> diameterASPL(box &p){
            int diameter=0;
            int64_t total=0;
            std::vector<std::vector<int>> edges(p.N);
            for(int x=0;x<p.Nx;x++){
                for(int y=0;y<p.Ny;y++){
                    for(int z=0;z<p.Nz;z++){
                        for(auto diff:p.edges[x%p.Mx][y%p.My][z%p.Mz]){
                            int dx,dy,dz;
                            std::tie(dx,dy,dz)=diff;
                            int tox = (x+dx+p.Nx)%p.Nx;
                            int toy = (y+dy+p.Ny)%p.Ny;
                            int toz = (z+dz+p.Nz)%p.Nz;
                            assert(p.xyz2id(tox,toy,toz)<p.N);
                            assert(p.xyz2id(x,y,z)<p.N);
                            edges[p.xyz2id(x,y,z)].push_back(p.xyz2id(tox,toy,toz));
                        }
                    }
                }
            }
            bool unconected = false;
            //#pragma omp parallel for reduction(+:total)
            for(int x=0;x<p.Mx;x++)for(int y=0;y<p.My;y++)for(int z=0;z<p.Mz;z++){
                int i = p.xyz2id(x,y,z);
                arrayQueue<int,capacity> arr1,arr2;
                uint64_t visited[capacity/64+1];
                for(int j=0;j<capacity/64+1;j++) visited[j]=0;
                auto *cur= &arr1;
                auto *next = &arr2;
                cur->clear(); next->clear();
                visited[i>>6]|=(1LL<<(i&0x3F));
                cur->push(i);
                int cnt=0;
                for(int step=0;!cur->empty();step++){
                    while(!cur->empty()){
                        int v=cur->front();cur->pop();
                        cnt++;
                        total+=step;
                        for(auto to:edges[v]){
                            if(visited[to>>6]&(1LL<<(to&0x3F))) continue;
                            visited[to>>6]|=(1LL<<(to&0x3F));
                            next->push(to);
                        }
                    }
                    if(next->empty()) diameter=std::max(diameter,step);
                    std::swap(cur,next);
                    next->clear();
                }
                if(cnt!=p.N) unconected=true;
            }
            if(unconected) diameter=std::numeric_limits<int>::max();
            total*=p.N/p.M;
            double aspl = double(total)/(int64_t(p.N)*(p.N-1));
            return std::make_pair(diameter,aspl);
        }
    };
}

