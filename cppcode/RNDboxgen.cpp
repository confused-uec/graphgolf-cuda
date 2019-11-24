#include <algorithm>
#include <vector>
#include <iostream>
#include <random>
#include <boost/program_options.hpp>
#include <fstream>
#include <box.hpp>
using namespace std;

int main(int argc, char* argv[]){
    boost::program_options::options_description opt("オプション");
    opt.add_options()
    ("help,h", "ヘルプを表示")
    ("nx", boost::program_options::value<int>()->default_value(10), "x軸頂点数")
    ("ny", boost::program_options::value<int>()->default_value(10), "y軸頂点数")
    ("nz", boost::program_options::value<int>()->default_value(10), "z軸頂点数")
    ("degree,d", boost::program_options::value<int>()->default_value(16), "最大次数")
    ("mx", boost::program_options::value<int>()->default_value(20), "1周期のx軸頂点数")
    ("my", boost::program_options::value<int>()->default_value(20), "1周期のy軸頂点数")
    ("mz", boost::program_options::value<int>()->default_value(20), "1周期のz軸頂点数")
    ("seed,s", boost::program_options::value<unsigned int>(), "乱数シード")
    ("output,o",boost::program_options::value<std::string>(), "output filepath")
    ("log,l",boost::program_options::value<std::string>(), "log filepath")
    ("verbose,v",boost::program_options::value<std::string>(), "verbose output filepath");
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


    int Nx=1,Ny=1,Nz=1,D=1,Mx=1,My=1,Mz=1;
    if(vm.count("nx")){
        Nx=vm["nx"].as<int>();
    }
    if(vm.count("ny")){
        Ny=vm["ny"].as<int>();
    }
    if(vm.count("nz")){
        Nz=vm["nz"].as<int>();
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
    if(vm.count("mz")){
        Mz=vm["mz"].as<int>();
    }

    vector<std::tuple<int,int,int>> v;
    for(int x=0;x<Mx;x++){
        for(int y=0;y<My;y++){
            for(int z=0;z<Mz;z++){
                for(int d=0;d<D;d++){
                    v.emplace_back(x,y,z);
                }
            }
        }
    }
    unsigned int seed;
    if(vm.count("seed")){
        seed=vm["seed"].as<unsigned int>();
    }else{
        random_device rd;
        seed=rd();
    }
    if(logging){
        if(vm.count("output")){
            logfs<<"output file: "<<vm["output"].as<std::string>()<<endl;
        }
        logfs<<"degree: "<<D<<endl;
        logfs<<"numberx: "<<Nx<<endl;
        logfs<<"numbery: "<<Ny<<endl;
        logfs<<"numberz: "<<Nz<<endl;
        logfs<<"modulex: "<<Mx<<endl;
        logfs<<"moduley: "<<My<<endl;
        logfs<<"moduley: "<<Mz<<endl;
        logfs<<"seed: "<<seed<<endl;
    }
    mt19937 engine(seed);
    shuffle(v.begin(), v.end(), engine);
    uniform_int_distribution<> distx(0,Nx/Mx-1),disty(0,Ny/My-1),distz(0,Nz/Mz-1);
    graphgolf::box p(Nx,Ny,Nz,Mx,My,Mz);
    for(int i=0;i<v.size();i+=2){
        int ax,ay,az,bx,by,bz;
        std::tie(ax,ay,az)=v[i];
        std::tie(bx,by,bz)=v[i+1];
        int diffx=(bx-ax)+Mx*distx(engine);
        int diffy=(by-ay)+My*disty(engine);
        int diffz=(bz-az)+Mz*distz(engine);
        p.edges[ax][ay][az].emplace_back(diffx,diffy,diffz);
        p.edges[bx][by][bz].emplace_back(-diffx,-diffy,-diffz);
    }
    std::basic_ostream<char> &ost = (outputfs.is_open()?outputfs:std::cout);
    p.print(ost);

    return 0;
}