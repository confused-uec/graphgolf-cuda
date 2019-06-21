
#include <iostream>
#include <boost/program_options.hpp>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include "part.hpp"
#include "cudaASPLconv.hpp"
#include "cudaASPLbeamer.hpp"

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
    graphgolf::part p;
    p.load(ist);
    if(logging){
        logfs<<"N: "<<p.N<<" M: "<<p.M<<std::endl;
    }

    int device=0;
    if(vm.count("device")) device=vm["device"].as<int>();
    //graphgolf::cudaASPLconv cu(p.N,p.M,p.degree,device);
    graphgolf::cudaASPLbeamer cu(p.N,p.M,p.degree,device);
    auto start = std::chrono::steady_clock::now();
    double aspl=cu.calc(p);
    auto end = std::chrono::steady_clock::now();
    std::cout.precision(11);
    ost.precision(11);
    std::cout<<"ASPL: "<<aspl<<std::endl;
    ost<<"ASPL: "<<aspl<<std::endl;
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<"TIME: "<<elapsed<<" msec."<<std::endl;
    ost<<"TIME: "<<elapsed<<" msec."<<std::endl;
    // return 0;
//    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
//    std::cout.precision(10);
//    std::cout<<"ASPL: "<<calcASPL<<std::endl;
//    std::cout<<"TIME: "<<elapsed<<" msec."<<std::endl;
    // int degree=16;
    // int64_t traversedEdges = int64_t(p.N)*p.M*p.degree;
    //std::cout<<"BFS Performance = "<<double(traversedEdges)/(elapsed/1000)/1000000<<" MTEPS"<<std::endl;
}