#include <bitset>
#include <array>
#include <vector>
#include <limits>
#include "arrayQueue.hpp"
#include "piece.hpp"
#include <tuple>
#include <cassert>
namespace graphgolf{
    template<int capacity> class cpuASPLpiece{
    public:
        std::pair<int,double> diameterASPL(piece &p){
            int diameter=0;
            int64_t total=0;
            std::vector<std::vector<int>> edges(p.Nx*p.Ny);
            for(int x=0;x<p.Nx;x++){
                for(int y=0;y<p.Ny;y++){
                    for(auto diff:p.edges[x%p.Mx][y%p.My]){
                        int dx,dy;
                        std::tie(dx,dy)=diff;
                        int tox = (x+dx+p.Nx)%p.Nx;
                        int toy = (y+dy+p.Ny)%p.Ny;
                        assert(tox*p.Ny+toy<p.Nx*p.Ny);
                        assert(x*p.Ny+y<p.Nx*p.Ny);
                        edges[x*p.Ny+y].push_back(tox*p.Ny+toy);
                    }
                }
            }
            bool unconected = false;
            //#pragma omp parallel for reduction(+:total)
            for(int x=0;x<p.Mx;x++)for(int y=0;y<p.My;y++){
                int i = x*p.Ny+y;
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
                if(cnt!=p.Nx*p.Ny) unconected=true;
            }
            if(unconected) diameter=std::numeric_limits<int>::max();
            total*=(p.Nx*p.Ny)/(p.Mx*p.My);
            double aspl = double(total)/(int64_t(p.Nx*p.Ny)*(p.Nx*p.Ny-1));
            return std::make_pair(diameter,aspl);
        }
    };
}

