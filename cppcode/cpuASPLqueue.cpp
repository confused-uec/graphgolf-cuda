#include <bitset>
#include <array>
#include <vector>
#include <limits>
#include "arrayQueue.hpp"
#include "part.hpp"

namespace graphgolf{
    template<int capacity> class cpuASPLqueue{
    public:
        double calc(part &p){
            int64_t total=0;
            std::vector<std::vector<int>> edges(p.N);
            for(int i=0;i<p.N;i++){
                for(auto diff:p.edges[i%p.M]){
                    edges[i].push_back((i+diff+p.N)%p.N);
                }
            }
            bool unconected = false;
            //#pragma omp parallel for reduction(+:total)
            for(int i=0;i<p.M;i++){
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
                    std::swap(cur,next);
                    next->clear();
                }
                if(cnt!=p.N){
                    unconected=true;
                    break;
                }
            }
            if(unconected) return 1e9;
            total*=p.N/p.M;
            return double(total)/(int64_t(p.N)*(p.N-1));
        }
        std::pair<int,double> diameterASPL(part &p){
            int diameter=0;
            int64_t total=0;
            std::vector<std::vector<int>> edges(p.N);
            for(int i=0;i<p.N;i++){
                for(auto diff:p.edges[i%p.M]){
                    edges[i].push_back((i+diff+p.N)%p.N);
                }
            }
            bool unconected = false;
            //#pragma omp parallel for reduction(+:total)
            for(int i=0;i<p.M;i++){
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

