#include <iostream>
#include <boost/program_options.hpp>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <tuple>
#include <iomanip>
#include "graph.hpp"
#include "cudaASPLgraph.hpp"

int main(int argc, char* argv[]){
    boost::program_options::options_description opt("オプション");
    opt.add_options()
    ("help,h", "ヘルプを表示")
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
        if(vm.count("input"))logfs<<"input file: "<<vm["input"].as<std::string>()<<std::endl;
        if(vm.count("output"))logfs<<"output file: "<<vm["output"].as<std::string>()<<std::endl;
    }
    graphgolf::graph g;
    g.load(ist);
    std::cout<<"Nodes: "<<g.N<<" Degrees_max: "<<g.degree_max<<std::endl;
    if(logging){
        logfs<<"Nodes: "<<g.N<<" Degrees_max: "<<g.degree_max<<std::endl;
    }
    int device=0;
    if(vm.count("device")) device=vm["device"].as<int>();
    graphgolf::cudaASPLgraph cu(g.N,g.degree_max,device);
    auto start = std::chrono::steady_clock::now();
    int diameter;
    int64_t total;
    std::tie(diameter,total)=cu.calc(g);
    double aspl= double(total)/(int64_t(g.N-1)*g.N);
    auto end = std::chrono::steady_clock::now();
    std::cout<<"Diameter: "<<diameter<<std::endl;
    ost<<"Diameter: "<<diameter<<std::endl;
    std::cout.precision(11);
    ost.precision(11);
    std::cout<<"ASPL: "<<aspl<<" ("<<total/2<<"/"<<int64_t(g.N)*(g.N-1)/2<<")"<<std::endl;
    ost<<"ASPL: "<<aspl<<std::endl;
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<"TIME: "<<elapsed<<" msec."<<std::endl;
    ost<<"TIME: "<<elapsed<<" msec."<<std::endl;
    return 0;
}