#include <iostream>
#include <functional>
#include <chrono>
#include <algorithm>
#include <boost/program_options.hpp>
#include <fstream>
#include <vector>
#include <random>
#include "box.hpp"
#include "cudaASPLbox.hpp"
#include <cassert>
int main(int argc, char* argv[]){
    auto init_time = std::chrono::steady_clock::now();
    boost::program_options::options_description opt("オプション");
    opt.add_options()
    ("help,h", "ヘルプを表示")
    ("count,c", boost::program_options::value<int>()->default_value(1000), "iteration count")
    ("seed,s", boost::program_options::value<unsigned int>(), "乱数シード")
	("input,i",boost::program_options::value<std::string>(), "input filepath")
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
    if(vm.count("input")){
        auto filename = vm["input"].as<std::string>();
        inputfs.open(filename, std::ios::binary | std::ios::in);
        if(!inputfs.is_open()){
            std::cerr<<"error: "<<filename<<" is not open"<<std::endl;
            exit(1);
        }
    }
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
        logfs.precision(10);
        logging=true;
    }
    if(vm.count("verbose")){
        auto filename = vm["verbose"].as<std::string>();
        verbosefs.open(filename, std::ios::binary | std::ios::out);
        if(!verbosefs.is_open()){
            std::cerr<<"error: "<<filename<<" is not open"<<std::endl;
            exit(1);
        }
        verbosefs.precision(10);
        verbose=true;
    }
    std::istream &ist = inputfs.is_open()?inputfs:std::cin;
    std::ostream &ost = outputfs.is_open()?outputfs:std::cout;
    if(verbose){
        if(vm.count("input"))verbosefs<<"#input file: "<<vm["input"].as<std::string>()<<std::endl;
        if(vm.count("output"))verbosefs<<"#output file: "<<vm["output"].as<std::string>()<<std::endl;
    }
    if(logging){
        if(vm.count("input"))logfs<<"#input file: "<<vm["input"].as<std::string>()<<std::endl;
        if(vm.count("output"))logfs<<"#output file: "<<vm["output"].as<std::string>()<<std::endl;
    }
    graphgolf::box init;
    init.load(ist);//初期解読み込み
    if(verbose){
        verbosefs<<"#Nx: "<<init.Nx<<"#Ny: "<<init.Ny<<"#Nz: "<<init.Nz
        <<" Mx: "<<init.Mx<<" My: "<<init.My<<" Mz: "<<init.Mz<<std::endl;
    }
    unsigned int seed;
    if(vm.count("seed")){
        seed=vm["seed"].as<unsigned int>();
    }else{
        std::random_device rd;
        seed=rd();
    }
    if(verbose){
        verbosefs<<"#seed: "<<seed<<std::endl;
    }
    std::mt19937 engine(seed);
    std::uniform_int_distribution<> dist_v(0,1);
    std::uniform_int_distribution<> dist_mx(0,init.Mx-1);
    std::uniform_int_distribution<> dist_my(0,init.My-1);
    std::uniform_int_distribution<> dist_mz(0,init.Mz-1);
    std::uniform_int_distribution<> dist_kx(0,init.Nx/init.Mx-1);
    std::uniform_int_distribution<> dist_ky(0,init.Ny/init.My-1);
    std::uniform_int_distribution<> dist_kz(0,init.Nz/init.Mz-1);
    std::function<graphgolf::box(graphgolf::box)> createNeighbour = [&](graphgolf::box p){
        int v = dist_v(engine);
        if((p.Nx/p.Mx!=1)&&(p.Ny/p.My-1!=0)&&(p.Nz/p.Mz-1!=0)&&v==0){
            //辺の長さを変化させる
            int fromx = dist_mx(engine);
            int fromy = dist_my(engine);
            int fromz = dist_mz(engine);
            std::uniform_int_distribution<> dist_e(0,p.edges[fromx][fromy][fromz].size()-1);
            int idx_from = dist_e(engine);
            int diffx,diffy,diffz;
            std::tie(diffx,diffy,diffz)=p.edges[fromx][fromy][fromz][idx_from];
            int tox = (p.Nx+fromx+diffx)%p.Mx;
            int toy = (p.Ny+fromy+diffy)%p.My;
            int toz = (p.Nz+fromz+diffz)%p.Mz;
            int idx_to=10000000;
            for(int i=0;i<p.edges[tox][toy][toz].size();i++){
                int tx,ty,tz;
                std::tie(tx,ty,tz)=p.edges[tox][toy][toz][i];
                if(tx+diffx==0&&ty+diffy==0&&tz+diffz==0){
                    if(diffx==0&&diffy==0&&diffz==0&&i==idx_from) continue;
                    idx_to=i;
                    break;
                }
            }
            int newdiffx=(tox-fromx)+p.Mx*dist_kx(engine);
            int newdiffy=(toy-fromy)+p.My*dist_ky(engine);
            int newdiffz=(toz-fromz)+p.Mz*dist_kz(engine);
            assert(tox<p.Mx&&toy<p.My&&toz<p.Mz);
            assert(fromx<p.Mx&&fromy<p.My&&fromz<p.Mz);
            assert(idx_from<p.edges[fromx][fromy][fromz].size());
            assert(idx_to<p.edges[tox][toy][toz].size());
            p.edges[fromx][fromy][fromz][idx_from]=std::make_tuple(newdiffx,newdiffy,newdiffz);
            p.edges[tox][toy][toz][idx_to]=std::make_tuple(-newdiffx,-newdiffy,-newdiffz);
            return p; 
        }else{
            //２本の辺を消して、交差
            // a---b, c---d -> a---d, b---cと貼り直す
            while(true){
                int ax=dist_mx(engine),ay=dist_my(engine),az=dist_mz(engine);
                std::uniform_int_distribution<> dist_eA(0,p.edges[ax][ay][az].size()-1);
                int idx_a=dist_eA(engine);
                int diff_abx,diff_aby,diff_abz;
                std::tie(diff_abx,diff_aby,diff_abz)=p.edges[ax][ay][az][idx_a];
                int bx=(p.Nx+ax+diff_abx)%p.Mx;
                int by=(p.Ny+ay+diff_aby)%p.My;
                int bz=(p.Nz+az+diff_abz)%p.Mz;
                int idx_b=1000000;
                for(int i=0;i<p.edges[bx][by][bz].size();i++){
                    int tx,ty,tz;
                    std::tie(tx,ty,tz)=p.edges[bx][by][bz][i];
                    if(tx+diff_abx==0&&ty+diff_aby==0&&tz+diff_abz==0){
                        if(diff_abx==0&&diff_aby==0&&diff_abz==0&&i==idx_a) continue;
                        idx_b=i;
                        break;
                    }
                }
                int cx=dist_mx(engine),cy=dist_my(engine),cz=dist_mz(engine);
                std::uniform_int_distribution<> dist_eC(0,p.edges[cx][cy][cz].size()-1);
                int idx_c=dist_eC(engine);
                int diff_cdx,diff_cdy,diff_cdz;
                std::tie(diff_cdx,diff_cdy,diff_cdz)=p.edges[cx][cy][cz][idx_c];
                int dx=(p.Nx+cx+diff_cdx)%p.Mx;
                int dy=(p.Ny+cy+diff_cdy)%p.My;
                int dz=(p.Nz+cz+diff_cdz)%p.Mz;
                int idx_d=1000000;
                for(int i=0;i<p.edges[dx][dy][dz].size();i++){
                    int tx,ty,tz;
                    std::tie(tx,ty,tz)=p.edges[dx][dy][dz][i];
                    if(tx+diff_cdx==0&&ty+diff_cdy==0&&tz+diff_cdz==0){
                        if(diff_cdx==0&&diff_cdy==0&&diff_cdz==0&&i==idx_c) continue;
                        idx_d=i;
                        break;
                    }
                }
                if(std::min(ax,bx)==std::min(cx,dx)&&
                   std::min(ay,by)==std::min(cy,dy)&&
                   std::min(az,bz)==std::min(cz,dz)&&
                   std::max(ax,bx)==std::max(cx,dx)&&
                   std::max(ay,by)==std::max(cy,dy)&&
                   std::max(az,bz)==std::max(cz,dz)&&
                   std::abs(diff_abx)==std::abs(diff_cdx)&&
                   std::abs(diff_aby)==std::abs(diff_cdy)&&
                   std::abs(diff_abz)==std::abs(diff_cdz)){
                    continue;//同じ辺を選んでしまうケースを回避(若干条件が緩い)
                } 
                int diff_adx=(dx-ax)+p.Mx*dist_kx(engine);
                int diff_ady=(dy-ay)+p.My*dist_ky(engine);
                int diff_adz=(dz-az)+p.Mz*dist_kz(engine);
                assert(ax<p.Mx&&ay<p.My&&az<p.Mz);
                assert(bx<p.Mx&&by<p.My&&bz<p.Mz);
                assert(cx<p.Mx&&cy<p.My&&cz<p.Mz);
                assert(dx<p.Mx&&dy<p.My&&dz<p.Mz);
                assert(idx_a<p.edges[ax][ay][az].size());
                assert(idx_d<p.edges[dx][dy][dz].size());
                p.edges[ax][ay][az][idx_a]=std::make_tuple(diff_adx,diff_ady,diff_adz);
                p.edges[dx][dy][dz][idx_d]=std::make_tuple(-diff_adx,-diff_ady,-diff_adz);
                int diff_bcx=(cx-bx)+p.Mx*dist_kx(engine);
                int diff_bcy=(cy-by)+p.My*dist_ky(engine);
                int diff_bcz=(cz-bz)+p.Mz*dist_kz(engine);
                assert(idx_b<p.edges[bx][by][bz].size());
                assert(idx_c<p.edges[cx][cy][cz].size());
                p.edges[bx][by][bz][idx_b]=std::make_tuple(diff_bcx,diff_bcy,diff_bcz);
                p.edges[cx][cy][cz][idx_c]=std::make_tuple(-diff_bcx,-diff_bcy,-diff_bcz);
                break;
            }
            return p;
        }
    };
    int device=0;
    if(vm.count("device")) device=vm["device"].as<int>();
    graphgolf::cudaASPLbox cu(init.Nx,init.Ny,init.Nz,init.Mx,init.My,init.Mz,init.degree,device);

    std::cout.precision(10);

    //double init_ASPL = cu.calc(init);
    double init_ASPL;
    int init_diam;
    std::tie(init_diam,init_ASPL)=cu.diameterASPL(init);
    graphgolf::box x = init;
    double fx=init_ASPL;
    int dx = init_diam;
    std::cout<<"ASPL(init_x): "<<init_ASPL<<std::endl;
    if(verbose){
        verbosefs<<"#ASPL(init_x): "<<init_ASPL<<std::endl;
    }
    if(logging){
        logfs<<"#iteration ASPL(x) ASPL(y)"<<std::endl;
        logfs<<0<<' '<<init_ASPL<<' '<<init_ASPL<<std::endl;
    }
    int count = 1000;
    if(vm.count("count")){
        count=vm["count"].as<int>();
    }
    for(int i=1;i<=count;i++){
        auto start = std::chrono::steady_clock::now();
        graphgolf::box y=createNeighbour(x);
        //double fy=cu.calc(y);
        double fy;
        int dy;
        std::tie(dy,fy)=cu.diameterASPL(y);
        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
        std::cout<<"iteration: "<<i<<" dx: "<<dx<<" fx: "<<fx<<" fy: "<<fy<<" time: "<<elapsed<<"ms"<<std::flush;
        if(verbose){
            verbosefs<<"iteration: "<<i<<" fx: "<<fx<<" fy: "<<fy<<std::endl;
        }
        if(logging){
            logfs<<i<<' '<<fx<<' '<<fy<<std::endl;
        }
        if(dy<dx||(dy==dx&&fy<fx)){
            std::cout<<std::endl;
            if(verbose){
                verbosefs<<"#update. new solution:"<<std::endl;
                y.print(verbosefs);
            }
            x=y;
            fx=fy;
            dx=dy;
        }
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-init_time).count();
    std::cout<<std::endl;
    std::cout<<"total elapsed time: "<<elapsed<<"s"<<std::endl; 
    if(verbosefs){
        verbosefs<<"#total elapsed time: "<<elapsed<<"s"<<std::endl; 
    }
    x.print(ost);
    return 0;
}