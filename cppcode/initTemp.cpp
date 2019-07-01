#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <boost/program_options.hpp>
#include <functional>
#include <chrono>
#include <fstream>
#include <cmath>
#include "part.hpp"
#include "cudaASPLconv.hpp"
#include "cpuASPLqueue.cpp"

int main(int argc, char* argv[]){
    boost::program_options::options_description opt("オプション");
    opt.add_options()
    ("help,h", "ヘルプを表示")
    ("number,n", boost::program_options::value<int>()->default_value(1000), "頂点数")
    ("degree,d", boost::program_options::value<int>()->default_value(16), "次数")
    ("module,m", boost::program_options::value<int>()->default_value(20), "1周期の頂点数")
    ("seed,s", boost::program_options::value<unsigned int>(), "乱数シード")
    ("output,o",boost::program_options::value<std::string>(), "output filepath")
    ("log,l",boost::program_options::value<std::string>(), "log filepath")
    ("verbose,v",boost::program_options::value<std::string>(), "verbose output filepath")
    ("device,D",boost::program_options::value<int>(), "GPU id");
    boost::program_options::variables_map vm;
    try{
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opt), vm);
        boost::program_options::notify(vm);
    }catch(const boost::program_options::error_with_option_name& e){
        std::cout << e.what() << std::endl;
        std::cout << opt << std::endl;
        exit(1);
    }
    
    //ヘルプを表示
    if(vm.count("help")){
        std::cout << opt << std::endl;
        exit(0);
    }
    
    std::ifstream inputfs;
    std::ofstream outputfs,verbosefs,logfs;
    bool verbose=false,logging=false;

    //ファイルを開く
    if(vm.count("output")){
        auto filename = vm["output"].as<std::string>();
        outputfs.open(filename, std::ios::binary | std::ios::out);
        if(!outputfs.is_open()){
            std::cerr<<"error: "<<filename<<" is not open"<<std::endl;
            exit(1);
        }
    }
    if(vm.count("log")){
        auto filename = vm["log"].as<std::string>();
        logfs.open(filename, std::ios::binary | std::ios::out);
        if(!logfs.is_open()){
            std::cerr<<"error: "<<filename<<" is not open"<<std::endl;
            exit(1);
        }
        logging=true;
    }
    if(vm.count("verbose")){
        auto filename = vm["verbose"].as<std::string>();
        verbosefs.open(filename, std::ios::binary | std::ios::out);
        if(!verbosefs.is_open()){
            std::cerr<<"error: "<<filename<<" is not open"<<std::endl;
            exit(1);
        }
        verbose=true;
    }


    int N=1000,D=16,M=20;
    if(vm.count("number")){
        N=vm["number"].as<int>();
    }
    if(vm.count("degree")){
        D=vm["degree"].as<int>();
    }
    if(vm.count("module")){
        M=vm["module"].as<int>();
    }

    std::vector<int> v(M*D);
    for(int i=0;i<M;i++){
        for(int j=0;j<D;j++){
            v[i*D+j]=i;
        }
    }
    unsigned int seed;
    if(vm.count("seed")){
        seed=vm["seed"].as<unsigned int>();
    }else{
        std::random_device rd;
        seed=rd();
    }
    if(logging){
        if(vm.count("output")){
            logfs<<"output file: "<<vm["output"].as<std::string>()<<std::endl;
        }
        logfs<<"degree: "<<D<<std::endl;
        logfs<<"number: "<<N<<std::endl;
        logfs<<"module: "<<M<<std::endl;
        logfs<<"seed: "<<seed<<std::endl;
    }
    std::mt19937 engine(seed);
    std::uniform_int_distribution<> dist(0,N/M-1);
    int device=0;
    if(vm.count("device")) device=vm["device"].as<int>();
    //graphgolf::cudaASPLconv cu(N,M,D,device);
    graphgolf::cpuASPLqueue<512> cu;
    double delta=0;
    int cnt=0;
    std::uniform_int_distribution<> dist_m(0,M-1);
    std::uniform_int_distribution<> dist_e(0,D-1);//正則を仮定
    std::uniform_int_distribution<> dist_v(0,1);
    std::uniform_int_distribution<> dist_k(0,N/M-1);
    std::function<graphgolf::part(graphgolf::part)> createNeighbour = [&](graphgolf::part p){
        int v = dist_v(engine);
        if(N/M-1!=0&&v==0){
            //辺の長さを変化させる
            int from = dist_m(engine);
            int idx_from = dist_e(engine);
            int diff = p.edges[from][idx_from];
            int to = (p.N+from+diff)%p.M;
            int idx_to=10000000;
            for(int i=0;i<p.edges[to].size();i++){
                if(p.edges[to][i]+diff==0){
                    if(diff==0&&i==idx_from) continue;
                    idx_to=i;
                    break;
                }
            }
            if(from>to){
                std::swap(from,to);
                std::swap(idx_from,idx_to);
            }
            int newdiff=(to-from)+p.M*dist_k(engine);
            assert(to<p.edges.size());
            assert(from<p.edges.size());
            assert(idx_from<p.edges[from].size());
            assert(idx_to<p.edges[to].size());
            p.edges[from][idx_from]=newdiff;
            p.edges[to][idx_to]=-newdiff;
            return p; 
        }else{
            //２本の辺を消して、交差
            // a---b, c---d -> a---d, b---cと貼り直す
            while(true){
                int a=dist_m(engine);
                int idx_a=dist_e(engine);
                int diff_ab=p.edges[a][idx_a];
                int b=(p.N+a+diff_ab)%p.M;
                int idx_b=1000000;
                for(int i=0;i<p.edges[b].size();i++){
                    if(p.edges[b][i]+diff_ab==0){
                        if(diff_ab==0&&i==idx_a) continue;
                        idx_b=i;
                        break;
                    }
                }
                int c=dist_m(engine);
                int idx_c=dist_e(engine);
                int diff_cd=p.edges[c][idx_c];
                int d=(p.N+c+diff_cd)%p.M;
                int idx_d=1000000;
                for(int i=0;i<p.edges[d].size();i++){
                    if(p.edges[d][i]+diff_cd==0){
                        if(diff_cd==0&&i==idx_c) continue;
                        idx_d=i;
                        break;
                    }
                }
                if(std::min(a,b)==std::min(c,d)&&
                   std::max(a,b)==std::max(c,d)&&
                   std::abs(diff_ab)==std::abs(diff_cd)){
                    continue;
                } 
                if(a>d){
                    std::swap(a,d);
                    std::swap(idx_a,idx_d);
                }
                int diff_ad=(d-a)+p.M*dist_k(engine);
                assert(a<p.edges.size());
                assert(b<p.edges.size());
                assert(c<p.edges.size());
                assert(d<p.edges.size());
                assert(idx_a<p.edges[a].size());
                assert(idx_d<p.edges[d].size());
                p.edges[a][idx_a]=diff_ad;
                p.edges[d][idx_d]=-diff_ad;
                if(b>c){
                    std::swap(b,c);
                    std::swap(idx_b,idx_c);
                }
                int diff_bc=(c-b)+p.M*dist_k(engine);
                assert(idx_b<p.edges[b].size());
                assert(idx_c<p.edges[c].size());
                p.edges[b][idx_b]=diff_bc;
                p.edges[c][idx_c]=-diff_bc;
                break;
            }
            return p;
        }
    };
    for(int n_part=0;n_part<100;n_part++){
        shuffle(v.begin(), v.end(), engine);
        std::vector<std::vector<int>> edges(M);
        for(int i=0;i<v.size();i+=2){
            int a=v[i],b=v[i+1];
            if(a>b)std::swap(a,b);
            int diff=(b-a)+M*dist(engine);
            edges[a].push_back(diff);
            edges[b].push_back(-diff);
        }
        graphgolf::part x;
        x.edges=edges;
        x.N=N;
        x.M=M;
        x.degree=D;
        double fx=cu.calc(x);
        std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
        std::cout<<n_part+1<<'/'<<100<<std::flush;
        for(int n_neighbor=0;n_neighbor<100;n_neighbor++){
            graphgolf::part y=createNeighbour(x);
            double fy=cu.calc(y);
            if(fy>fx&&fy!=1e9){
                cnt++;
                delta+=fy-fx;
            }
        }
    }
    double avg=delta/cnt*N*(N-1);
    std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
    std::cout<<"cnt: "<<cnt<<std::endl;
    std::cout<<"avg: "<<avg<<std::endl;
    std::cout<<"inittemp: "<<-avg/log(0.4)<<std::endl;

    return 0;
}