cmake_minimum_required(VERSION 3.0.0)
project(cudatest VERSION 0.1.0)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

find_package(Boost REQUIRED COMPONENTS program_options filesystem)
include_directories(Boost::boost)
set(SOURCES cppcode/main.cpp)
set(HEADERS cppcode/arrayQueue.hpp cppcode/part.hpp cppcode/piece.hpp cppcode/box.hpp)

add_executable(main ${SOURCES} ${HEADERS})
target_link_libraries(main Boost::program_options Boost::filesystem)
add_subdirectory(cudacode)
target_include_directories(main PUBLIC cudacode cppcode)
target_link_libraries(main cudaASPL)

add_executable(HillClimb cppcode/HillClimb.cpp ${HEADERS})
target_include_directories(HillClimb PUBLIC cudacode cppcode)
target_link_libraries(HillClimb cudaASPL Boost::program_options)

add_executable(HillClimbWVC cppcode/HillClimbWVC.cpp ${HEADERS})
target_include_directories(HillClimbWVC PUBLIC cudacode cppcode)
target_link_libraries(HillClimbWVC cudaASPL Boost::program_options)

add_executable(aspl_cuda cppcode/aspl_cuda.cpp cppcode/graph.hpp)
target_include_directories(aspl_cuda PUBLIC cudacode cppcode)
target_link_libraries(aspl_cuda cudaASPLgraph Boost::program_options)

add_executable(beamer cppcode/beamer.cpp ${HEADERS})
target_include_directories(beamer PUBLIC cudacode cppcode)
target_link_libraries(beamer cudaASPL Boost::program_options)

add_executable(annealing cppcode/annealing.cpp ${HEADERS} cppcode/cpuASPLqueue.cpp)
target_include_directories(annealing PUBLIC cudacode cppcode)
target_link_libraries(annealing cudaASPL Boost::program_options)

add_executable(annealingNR cppcode/annealingNR.cpp ${HEADERS} cppcode/cpuASPLqueue.cpp)
target_include_directories(annealingNR PUBLIC cudacode cppcode)
target_link_libraries(annealingNR cudaASPL Boost::program_options)

add_executable(annealingEXP cppcode/annealingEXP.cpp ${HEADERS} cppcode/cpuASPLqueue.cpp)
target_include_directories(annealingEXP PUBLIC cudacode cppcode)
target_link_libraries(annealingEXP cudaASPL Boost::program_options)

add_executable(annealingWVC cppcode/annealingWVC.cpp ${HEADERS} cppcode/cpuASPLqueue.cpp)
target_include_directories(annealingWVC PUBLIC cudacode cppcode)
target_link_libraries(annealingWVC cudaASPL Boost::program_options)

add_executable(annealing_cpu cppcode/annealing_cpu.cpp ${HEADERS} cppcode/cpuASPLqueue.cpp)
target_include_directories(annealing_cpu PUBLIC cudacode cppcode)
target_link_libraries(annealing_cpu Boost::program_options)

add_executable(annealingEXP_cpu cppcode/annealingEXP_cpu.cpp ${HEADERS} cppcode/cpuASPLqueue.cpp)
target_include_directories(annealingEXP_cpu PUBLIC cudacode cppcode)
target_link_libraries(annealingEXP_cpu Boost::program_options)

add_executable(HillClimb_cpu cppcode/HillClimb_cpu.cpp ${HEADERS} cppcode/cpuASPLqueue.cpp)
target_include_directories(HillClimb_cpu PUBLIC cudacode cppcode)
target_link_libraries(HillClimb_cpu Boost::program_options)

add_executable(initTemp cppcode/initTemp.cpp ${HEADERS} cppcode/cpuASPLqueue.cpp)
target_include_directories(initTemp PUBLIC cudacode cppcode)
target_link_libraries(initTemp cudaASPL Boost::program_options)

add_executable(RNDpiecegen cppcode/RNDpiecegen.cpp ${HEADERS})
target_include_directories(RNDpiecegen PUBLIC cudacode cppcode)
target_link_libraries(RNDpiecegen Boost::program_options)

add_executable(piece2graph cppcode/piece2graph.cpp ${HEADERS})
target_include_directories(piece2graph PUBLIC cudacode cppcode)
target_link_libraries(piece2graph Boost::program_options)

add_executable(piece2aspl cppcode/piece2aspl.cpp ${HEADERS} cppcode/cpuASPLpiece.cpp)
target_include_directories(piece2aspl PUBLIC cudacode cppcode)
target_link_libraries(piece2aspl cudaASPL Boost::program_options Boost::filesystem)

add_executable(pieceHillClimb cppcode/pieceHillClimb.cpp ${HEADERS})
target_include_directories(pieceHillClimb PUBLIC cudacode cppcode)
target_link_libraries(pieceHillClimb cudaASPL Boost::program_options)

add_executable(pieceHillClimb_cpu cppcode/pieceHillClimb_cpu.cpp ${HEADERS} cppcode/cpuASPLpiece.cpp)
target_include_directories(pieceHillClimb_cpu PUBLIC cudacode cppcode)
target_link_libraries(pieceHillClimb_cpu cudaASPL Boost::program_options)

add_executable(pieceinitTemp cppcode/pieceinitTemp.cpp ${HEADERS} cppcode/cpuASPLpiece.cpp)
target_include_directories(pieceinitTemp PUBLIC cudacode cppcode)
target_link_libraries(pieceinitTemp cudaASPL Boost::program_options)

add_executable(pieceannealing cppcode/pieceannealing.cpp ${HEADERS})
target_include_directories(pieceannealing PUBLIC cudacode cppcode)
target_link_libraries(pieceannealing cudaASPL Boost::program_options)

add_executable(pieceannealing_cpu cppcode/pieceannealing_cpu.cpp ${HEADERS} cppcode/cpuASPLpiece.cpp)
target_include_directories(pieceannealing_cpu PUBLIC cudacode cppcode)
target_link_libraries(pieceannealing_cpu cudaASPL Boost::program_options)

add_executable(RNDboxgen cppcode/RNDboxgen.cpp ${HEADERS})
target_include_directories(RNDboxgen PUBLIC cudacode cppcode)
target_link_libraries(RNDboxgen Boost::program_options)

add_executable(box2aspl cppcode/box2aspl.cpp ${HEADERS} cppcode/cpuASPLbox.cpp)
target_include_directories(box2aspl PUBLIC cudacode cppcode)
target_link_libraries(box2aspl cudaASPL Boost::program_options Boost::filesystem)

add_executable(box2graph cppcode/box2graph.cpp ${HEADERS})
target_include_directories(box2graph PUBLIC cudacode cppcode)
target_link_libraries(box2graph Boost::program_options)

add_executable(boxHillClimb cppcode/boxHillClimb.cpp ${HEADERS})
target_include_directories(boxHillClimb PUBLIC cudacode cppcode)
target_link_libraries(boxHillClimb cudaASPL Boost::program_options)

#find_package(OpenMP REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
