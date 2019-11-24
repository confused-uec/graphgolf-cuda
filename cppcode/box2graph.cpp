#include <iostream>
#include <cmath>
#include <boost/program_options.hpp>
#include <fstream>
#include <vector>
#include <box.hpp>

int main(int argc, char* argv[]){
    boost::program_options::options_description opt("オプション");
    opt.add_options()
    ("help,h", "ヘルプを表示")
    ("seed,s", boost::program_options::value<unsigned int>(), "乱数シード")
	("input,i",boost::program_options::value<std::string>(), "input filepath")
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

    std::istream &ist = inputfs.is_open()?inputfs:std::cin;
    std::ostream &ost = outputfs.is_open()?outputfs:std::cout;
    if(logging){
        logfs<<"input file: "<<vm["input"].as<std::string>()<<std::endl;
        logfs<<"output file: "<<vm["output"].as<std::string>()<<std::endl;
    }
    graphgolf::box p;
    p.load(ist);
    if(logging){
        logfs<<"Nx: "<<p.Nx<<" Ny: "<<p.Ny<<" Nz: "<<p.Nz
            <<" Mx: "<<p.Mx<<" My: "<<p.My<<" Mz: "<<p.Mz<<std::endl;
    }
    std::vector<std::pair<int,int>> V;
    for(int x=0;x<p.Nx;x++){
        for(int y=0;y<p.Ny;y++){
            for(int z=0;z<p.Nz;z++){
                for(auto e:p.edges[x%p.Mx][y%p.My][z%p.Mz]){
                    int diffx,diffy,diffz;
                    std::tie(diffx,diffy,diffz)=e;
                    int a=p.xyz2id(x,y,z);
                    int b=p.xyz2id((x+diffx+p.Nx)%p.Nx,(y+diffy+p.Ny)%p.Ny,(z+diffz+p.Nz)%p.Nz);
                    if(a>b)std::swap(a,b);
                    V.emplace_back(a,b);
                }
            }
        }
    }
    std::sort(V.begin(), V.end());
    V.erase(std::unique(V.begin(),V.end()),V.end());
    for(auto p:V){
        int a,b;
        std::tie(a,b)=p;
        ost<<a<<' '<<b<<std::endl;
    }
}