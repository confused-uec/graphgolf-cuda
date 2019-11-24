#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <boost/program_options.hpp>
#include <functional>
#include <chrono>
#include <fstream>
#include <cmath>
#include "piece.hpp"
#include "cudaASPLpiece.hpp"
#include "cpuASPLpiece.cpp"

int main(int argc, char* argv[]){
    boost::program_options::options_description opt("オプション");
    opt.add_options()
    ("help,h", "ヘルプを表示")
    ("nx", boost::program_options::value<int>()->default_value(10), "x軸頂点数")
    ("ny", boost::program_options::value<int>()->default_value(10), "y軸頂点数")
    ("degree,d", boost::program_options::value<int>()->default_value(16), "最大次数")
    ("mx", boost::program_options::value<int>()->default_value(20), "1周期のx軸頂点数")
    ("my", boost::program_options::value<int>()->default_value(20), "1周期のy軸頂点数")
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


    int Nx=1,Ny=1,D=1,Mx=1,My=1;
    if(vm.count("nx")){
        Nx=vm["nx"].as<int>();
    }
    if(vm.count("ny")){
        Ny=vm["ny"].as<int>();
    }
    if(vm.count("degree")){
        D=vm["degree"].as<int>();
    }
    if(vm.count("mx")){
        Mx=vm["mx"].as<int>();
    }
    if(vm.count("my")){
        My=vm["my"].as<int>();
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
        logfs<<"numberx: "<<Nx<<std::endl;
        logfs<<"numbery: "<<Ny<<std::endl;
        logfs<<"modulex: "<<Mx<<std::endl;
        logfs<<"moduley: "<<My<<std::endl;
        logfs<<"seed: "<<seed<<std::endl;
    }
    std::mt19937 engine(seed);
    std::uniform_int_distribution<> dist_v(0,1);
    std::uniform_int_distribution<> dist_mx(0,Mx-1);
    std::uniform_int_distribution<> dist_my(0,My-1);
    std::uniform_int_distribution<> dist_kx(0,Nx/Mx-1);
    std::uniform_int_distribution<> dist_ky(0,Ny/My-1);
    std::function<graphgolf::piece(graphgolf::piece)> createNeighbour = [&](graphgolf::piece p){
        int v = dist_v(engine);
        if((p.Nx/p.Mx!=1)&&(p.Ny/p.My-1!=0)&&v==0){
            //辺の長さを変化させる
            int fromx = dist_mx(engine);
            int fromy = dist_my(engine);
            std::uniform_int_distribution<> dist_e(0,p.edges[fromx][fromy].size()-1);
            int idx_from = dist_e(engine);
            int diffx,diffy;
            std::tie(diffx,diffy)=p.edges[fromx][fromy][idx_from];
            int tox = (p.Nx+fromx+diffx)%p.Mx;
            int toy = (p.Ny+fromy+diffy)%p.My;
            int idx_to=10000000;
            for(int i=0;i<p.edges[tox][toy].size();i++){
                if(p.edges[tox][toy][i].first+diffx==0&&p.edges[tox][toy][i].second+diffy==0){
                    if(diffx==0&&diffy==0&&i==idx_from) continue;
                    idx_to=i;
                    break;
                }
            }
            int newdiffx=(tox-fromx)+p.Mx*dist_kx(engine);
            int newdiffy=(toy-fromy)+p.My*dist_ky(engine);
            assert(tox<p.Mx&&toy<p.My);
            assert(fromx<p.Mx&&fromy<p.My);
            assert(idx_from<p.edges[fromx][fromy].size());
            assert(idx_to<p.edges[tox][toy].size());
            p.edges[fromx][fromy][idx_from]=std::make_pair(newdiffx,newdiffy);
            p.edges[tox][toy][idx_to]=std::make_pair(-newdiffx,-newdiffy);
            return p; 
        }else{
            //２本の辺を消して、交差
            // a---b, c---d -> a---d, b---cと貼り直す
            while(true){
                int ax=dist_mx(engine),ay=dist_my(engine);
                std::uniform_int_distribution<> dist_eA(0,p.edges[ax][ay].size()-1);
                int idx_a=dist_eA(engine);
                int diff_abx,diff_aby;
                std::tie(diff_abx,diff_aby)=p.edges[ax][ay][idx_a];
                int bx=(p.Nx+ax+diff_abx)%p.Mx;
                int by=(p.Ny+ay+diff_aby)%p.My;
                int idx_b=1000000;
                for(int i=0;i<p.edges[bx][by].size();i++){
                    if(p.edges[bx][by][i].first+diff_abx==0&&p.edges[bx][by][i].second+diff_aby==0){
                        if(diff_abx==0&&diff_aby==0&&i==idx_a) continue;
                        idx_b=i;
                        break;
                    }
                }
                int cx=dist_mx(engine),cy=dist_my(engine);
                std::uniform_int_distribution<> dist_eC(0,p.edges[cx][cy].size()-1);
                int idx_c=dist_eC(engine);
                int diff_cdx,diff_cdy;
                std::tie(diff_cdx,diff_cdy)=p.edges[cx][cy][idx_c];
                int dx=(p.Nx+cx+diff_cdx)%p.Mx;
                int dy=(p.Ny+cy+diff_cdy)%p.My;
                int idx_d=1000000;
                for(int i=0;i<p.edges[dx][dy].size();i++){
                    if(p.edges[dx][dy][i].first+diff_cdx==0&&p.edges[dx][dy][i].second+diff_cdy==0){
                        if(diff_cdx==0&&diff_cdy==0&&i==idx_c) continue;
                        idx_d=i;
                        break;
                    }
                }
                if(std::min(ax,bx)==std::min(cx,dx)&&
                   std::min(ay,by)==std::min(cy,dy)&&
                   std::max(ax,bx)==std::max(cx,dx)&&
                   std::max(ay,by)==std::max(cy,dy)&&
                   std::abs(diff_abx)==std::abs(diff_cdx)&&
                   std::abs(diff_abx)==std::abs(diff_cdx)){
                    continue;//同じ辺を選んでしまうケースを回避(若干条件が緩い)
                } 
                int diff_adx=(dx-ax)+p.Mx*dist_kx(engine);
                int diff_ady=(dy-ay)+p.My*dist_ky(engine);
                assert(ax<p.Mx&&ay<p.My);
                assert(bx<p.Mx&&by<p.My);
                assert(cx<p.Mx&&cy<p.My);
                assert(dx<p.Mx&&dy<p.My);
                assert(idx_a<p.edges[ax][ay].size());
                assert(idx_d<p.edges[dx][dy].size());
                p.edges[ax][ay][idx_a]=std::make_pair(diff_adx,diff_ady);
                p.edges[dx][dy][idx_d]=std::make_pair(-diff_adx,-diff_ady);
                int diff_bcx=(cx-bx)+p.Mx*dist_kx(engine);
                int diff_bcy=(cy-by)+p.My*dist_ky(engine);
                assert(idx_b<p.edges[bx][by].size());
                assert(idx_c<p.edges[cx][cy].size());
                p.edges[bx][by][idx_b]=std::make_pair(diff_bcx,diff_bcy);
                p.edges[cx][cy][idx_c]=std::make_pair(-diff_bcx,-diff_bcy);
                break;
            }
            return p;
        }
    };

    int device=0;
    if(vm.count("device")) device=vm["device"].as<int>();
    graphgolf::cudaASPLpiece cu(Nx,Ny,Mx,My,D,device);
    //graphgolf::cpuASPLpiece<65536> cu;
    double delta=0;
    int cnt=0;


    std::vector<std::pair<int,int>> v;
    for(int x=0;x<Mx;x++){
        for(int y=0;y<My;y++){
            for(int d=0;d<D;d++){
                v.emplace_back(x,y);
            }
        }
    }

    for(int n_part=0;n_part<100;n_part++){
        shuffle(v.begin(), v.end(), engine);
        std::uniform_int_distribution<> distx(0,Nx/Mx-1),disty(0,Ny/My-1);
        graphgolf::piece x(Nx,Ny,Mx,My);
        for(int i=0;i<v.size();i+=2){
            int ax,ay,bx,by;
            std::tie(ax,ay)=v[i];
            std::tie(bx,by)=v[i+1];
            int diffx=(bx-ax)+Mx*distx(engine);
            int diffy=(by-ay)+My*disty(engine);
            x.edges[ax][ay].emplace_back(diffx,diffy);
            x.edges[bx][by].emplace_back(-diffx,-diffy);
        }
        double fx;
        std::tie(std::ignore,fx)=cu.diameterASPL(x);
        std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
        std::cout<<n_part+1<<'/'<<100<<std::flush;
        for(int n_neighbor=0;n_neighbor<100;n_neighbor++){
            graphgolf::piece y=createNeighbour(x);
            double fy;
            std::tie(std::ignore,fy)=cu.diameterASPL(y);
            if(fy>fx&&fy!=1e9){
                cnt++;
                delta+=fy-fx;
            }
        }
    }
    int N=Nx*Ny;
    double avg=delta/cnt*N*(N-1);
    std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
    std::cout<<"cnt: "<<cnt<<std::endl;
    std::cout<<"avg: "<<avg<<std::endl;
    std::cout<<"inittemp: "<<-avg/log(0.4)<<std::endl;

    return 0;
}