#include <iostream>
#include <functional>
#include <chrono>
#include <algorithm>
#include <boost/program_options.hpp>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include "part.hpp"
#include "cudaASPLconv.hpp"
#include "cpuASPLqueue.cpp"

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
    ("device,D",boost::program_options::value<int>(), "GPU id")
    ("temp,t",boost::program_options::value<double>(),"initial templature");
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
    graphgolf::part init;
    init.load(ist);//初期解読み込み
    if(verbose){
        verbosefs<<"#N: "<<init.N<<" M: "<<init.M<<std::endl;
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
    std::uniform_int_distribution<> dist_m(0,init.M-1);
    //std::uniform_int_distribution<> dist_e(0,init.degree-1);//正則を仮定
    std::uniform_int_distribution<> dist_v(0,1);
    std::uniform_int_distribution<> dist_k(0,init.N/init.M-1);
    std::function<graphgolf::part(graphgolf::part)> createNeighbour = [&](graphgolf::part p){
        int v = dist_v(engine);
        if(init.N/init.M-1!=0&&v==0){
            //辺の長さを変化させる
            int from = dist_m(engine);
            std::uniform_int_distribution<> dist_e(0,p.edges[from].size()-1);
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
        }else if(v<=1){
            //２本の辺を消して、交差
            // a---b, c---d -> a---d, b---cと貼り直す
            while(true){
                int a=dist_m(engine);
                std::uniform_int_distribution<> dist_eA(0,p.edges[a].size()-1);
                int idx_a=dist_eA(engine);
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
                std::uniform_int_distribution<> dist_eC(0,p.edges[c].size()-1);
                int idx_c=dist_eC(engine);
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
        }else{
            //3本の辺を消して、交差
            // a-b, c-d, e-f -> f-a, b-c, d-eと貼り直す
            while(true){
                int a=dist_m(engine);
                std::uniform_int_distribution<> dist_eA(0,p.edges[a].size()-1);
                int idx_a=dist_eA(engine);
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
                std::uniform_int_distribution<> dist_eC(0,p.edges[c].size()-1);
                int idx_c=dist_eC(engine);
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
                int e=dist_m(engine);
                std::uniform_int_distribution<> dist_eE(0,p.edges[e].size()-1);
                int idx_e=dist_eE(engine);
                int diff_ef=p.edges[e][idx_e];
                int f=(p.N+e+diff_ef)%p.M;
                int idx_f=1000000;
                for(int i=0;i<p.edges[f].size();i++){
                    if(p.edges[f][i]+diff_ef==0){
                        if(diff_ef==0&&i==idx_e) continue;
                        idx_f=i;
                        break;
                    }
                }
                if(std::min(a,b)==std::min(c,d)&&
                   std::max(a,b)==std::max(c,d)&&
                   std::abs(diff_ab)==std::abs(diff_cd)){
                    continue;
                }
                if(std::min(c,d)==std::min(e,f)&&
                   std::max(c,d)==std::max(e,f)&&
                   std::abs(diff_cd)==std::abs(diff_ef)){
                    continue;
                }
                if(std::min(e,f)==std::min(a,b)&&
                   std::max(e,f)==std::max(a,b)&&
                   std::abs(diff_ef)==std::abs(diff_ab)){
                    continue;
                }
                
                assert(a<p.edges.size());
                assert(b<p.edges.size());
                assert(c<p.edges.size());
                assert(d<p.edges.size());
                assert(e<p.edges.size());
                assert(f<p.edges.size());
                
                if(f>a){
                    std::swap(f,a);
                    std::swap(idx_f,idx_a);
                }
                int diff_fa=(a-f)+p.M*dist_k(engine);
                assert(idx_a<p.edges[a].size());
                assert(idx_d<p.edges[d].size());
                p.edges[f][idx_f]=diff_fa;
                p.edges[a][idx_a]=-diff_fa;
                if(b>c){
                    std::swap(b,c);
                    std::swap(idx_b,idx_c);
                }
                int diff_bc=(c-b)+p.M*dist_k(engine);
                assert(idx_b<p.edges[b].size());
                assert(idx_c<p.edges[c].size());
                p.edges[b][idx_b]=diff_bc;
                p.edges[c][idx_c]=-diff_bc;

                if(d>e){
                    std::swap(d,e);
                    std::swap(idx_d,idx_e);
                }
                int diff_de=(e-d)+p.M*dist_k(engine);
                assert(idx_d<p.edges[d].size());
                assert(idx_e<p.edges[e].size());
                p.edges[d][idx_d]=diff_de;
                p.edges[e][idx_e]=-diff_de;
                break;
            }
            return p;
        }
    };
    //graphgolf::cpuASPLqueue<512> cu;
    int device=0;
    if(vm.count("device")) device=vm["device"].as<int>();
    graphgolf::cudaASPLconv cu(init.N,init.M,init.degree,device);
    std::cout.precision(10);

    //double init_ASPL = cu.calc(init);
    double init_ASPL;
    int init_Diameter;
    std::tie(init_Diameter,init_ASPL) = cu.diameterASPL(init);
    if(init_Diameter==100000000){
        if(logging){
            logfs<<"#initial solution is unconnected"<<std::endl;
        }
        std::cout<<"initial solution is unconnected"<<std::endl;
        return 0;
    }
    graphgolf::part x = init;
    double fx=init_ASPL;
    int dx=init_Diameter;
    graphgolf::part x_best = x;
    double fx_best = fx;
    int dx_best = dx;
    std::cout<<"ASPL(init_x): "<<init_ASPL<<std::endl;
    if(verbose){
        verbosefs<<"#ASPL(init_x): "<<init_ASPL<<std::endl;
    }
    //double temp = 0.001091346;
    double inittemp = 46.1793;
    if(vm.count("temp"))inittemp=vm["temp"].as<double>();
    if(logging){
        logfs<<"#iteration ASPL(x_best) ASPL(x) ASPL(y) temp"<<std::endl;
        logfs<<0<<' '<<init_ASPL<<' '<<init_ASPL<<' '<<init_ASPL<<' '<<inittemp<<std::endl;
    }
    int count = 1000;
    if(vm.count("count")){
        count=vm["count"].as<int>();
    }
    std::uniform_real_distribution<double> dist_p(0,1);
    /*
    {//実践的焼きなまし法
        
        const int NEIGHBOURSIZE = std::min(100000,(x.M*x.degree)*(x.M*x.degree)*(x.N/x.M)*(x.N/x.M));
        std::cout<<"NSIZE: "<<NEIGHBOURSIZE<<std::endl;
        const double TEMPFACTOR = 0.99;
        const int SIZEFACTOR = 16;
        const double MINPERCENT = 0.02;
        const int FREEZELIM = 5;
        const double CUTOFF = 0.1;
        double T = inittemp;
        int iter=0;
        for(int freeze=0,i=0;freeze<FREEZELIM;true){
            bool x_best_update=false;
            int changes, trials;
            for(changes=trials=0;trials<SIZEFACTOR*NEIGHBOURSIZE&&changes<CUTOFF*NEIGHBOURSIZE;trials++){
                auto start = std::chrono::steady_clock::now();
                graphgolf::part y=createNeighbour(x);
                double fy=cu.calc(y);
                auto end = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/1000.0;
                iter++;
                double delta = fy-fx;
                bool accept=false;
                if(fy<fx){
                    accept=true;
                }else if(dist_p(engine)<exp(-delta*x.N*(x.N-1)/T)){
                    accept=true;
                }
                if(accept) changes++;
                if(i%1000==0){
                        std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
                        std::cout<<"iter: "<<iter<<" fx_best: "<<fx_best<<" fx: "<<fx<<" temp: "<<T<<" time: "<<elapsed<<"ms"<<std::flush;
                }
                if(accept){
                    if(fx!=fy){
                        std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
                        std::cout<<"iter: "<<iter<<" fx_best: "<<fx_best<<" fx: "<<fx<<" temp: "<<T<<" time: "<<elapsed<<"ms"<<std::flush;
                        if(verbose){
                            verbosefs<<"iter: "<<iter<<" fx_best: "<<fx_best<<" fx: "<<fx<<" temp: "<<T<<std::endl;
                        }
                        if(logging) logfs<<iter<<' '<<fx_best<<' '<<fx<<' '<<T<<std::endl;
                    }
                    fx=fy;
                    x=y;
                }
                if(fy<fx_best){
                    std::cout<<std::endl;
                    if(verbose){
                        verbosefs<<"#update. new solution:"<<std::endl;
                        y.print(verbosefs);
                    }
                    x_best=y;
                    fx_best=fy;
                    x_best_update=true;
                }
            }
            T*=TEMPFACTOR;
            if(x_best_update)freeze=0;
            if(double(changes)/trials<MINPERCENT) freeze++;
            if(T<0.1)break;
        }
    }*/
    /*
    double temp = inittemp;
    for(int i=1;i<=count;i++){
        auto start = std::chrono::steady_clock::now();
        graphgolf::part y=createNeighbour(x);
        //double fy=cu.calc(y);
        double fy;
        int dy;
        std::tie(dy,fy)=cu.diameterASPL(y);
        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/1000.0;
        bool accept=false;
        if(fy<fx){
            accept=true;
        }else if(dist_p(engine)<exp((fx-fy)*x.N*(x.N-1)/temp)){
            accept=true;
        }
        if(dy>dx) accept = false;
        if(i%1000==0){
                std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
                std::cout<<"iteration: "<<i<<" fx_best: "<<fx_best<<" fx: "<<fx<<" time: "<<elapsed<<"ms"<<std::flush;
        }
        if(accept){
            if(fx!=fy){
                std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
                std::cout<<"iteration: "<<i<<" fx_best: "<<fx_best<<" fx: "<<fx<<" time: "<<elapsed<<"ms"<<std::flush;
                if(verbose){
                    verbosefs<<"iteration: "<<i<<" fx_best: "<<fx_best<<" fx: "<<fx<<" temp: "<<temp<<std::endl;
                }
                if(logging) logfs<<i<<' '<<fx_best<<' '<<fx<<' '<<temp<<std::endl;
            }
            fx=fy;
            dx=dy;
            x=y;
        }
        if(accept&&fy<fx_best&&dy<=dx_best){
            std::cout<<std::endl;
            if(verbose){
                verbosefs<<"#update. new solution:"<<std::endl;
                y.print(verbosefs);
            }
            x_best=y;
            fx_best=fy;
            dx_best=dy;
        }
        //if(i%10000==0) temp=inittemp*(std::tanh(double(count-i)/count*6-3)+1)/2;
        //if(i%10000==0) temp=inittemp*std::tanh(double(count-i)/count*3);
        if(i%10000==0) temp=inittemp*std::pow(0.995,i/10000);
    }
    */
    for(int i=1;i<=count;i++){
        auto start = std::chrono::steady_clock::now();
        graphgolf::part y=createNeighbour(x);
        //double fy=cu.calc(y);
        double fy;
        int dy;
        std::tie(dy,fy)=cu.diameterASPL(y);
        auto end = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()/1000.0;
        bool accept=false;
        double temp = inittemp*std::exp(-double(i)/count*std::log(inittemp));
        if(fy<fx||dy<dx){
            accept=true;
        }else if(dist_p(engine)<exp((fx-fy)*x.N*(x.N-1)/temp)){
            accept=true;
        }
        if(dy>dx) accept = false;
        if(i%1000==0){
                std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
                std::cout<<"iteration: "<<i<<" dx: "<<dx<<" fx_best: "<<fx_best<<" fx: "<<fx<<" time: "<<elapsed<<"ms"<<std::flush;
        }
        if(accept){
            if(fx!=fy){
                std::cout<<char(27)<<'['<<'F'<<char(27)<<'['<<'E'<<char(27)<<'['<<'K'<<std::flush;
                std::cout<<"iteration: "<<i<<" dx: "<<dx<<" fx_best: "<<fx_best<<" fx: "<<fx<<" time: "<<elapsed<<"ms"<<std::flush;
                if(verbose){
                    verbosefs<<"iteration: "<<i<<" fx_best: "<<fx_best<<" fx: "<<fx<<" temp: "<<temp<<std::endl;
                }
                if(logging) logfs<<i<<' '<<fx_best<<' '<<fx<<' '<<temp<<std::endl;
            }
            fx=fy;
            dx=dy;
            x=y;
        }
        if(dy<dx_best||(dy==dx_best&&fy<fx_best)){
            std::cout<<std::endl;
            if(verbose){
                verbosefs<<"#update. new solution:"<<std::endl;
                y.print(verbosefs);
            }
            x_best=y;
            fx_best=fy;
            dx_best=dy;
        }
        //if(i%10000==0) temp=inittemp*(std::tanh(double(count-i)/count*6-3)+1)/2;
        //if(i%10000==0) temp=inittemp*std::tanh(double(count-i)/count*3);
        //if(i%10000==0) temp=inittemp*std::pow(0.995,i/10000);
    }
       
    
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::seconds>(end-init_time).count();
    std::cout<<std::endl;
    std::cout<<"total elapsed time: "<<elapsed<<"s"<<std::endl; 
    if(verbosefs){
        verbosefs<<"#total elapsed time: "<<elapsed<<"s"<<std::endl; 
    }
    x_best.print(ost);
    return 0;
}