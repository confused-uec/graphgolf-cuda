cmake_minimum_required(VERSION 3.0.0)
project(cudatest VERSION 0.1.0)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(Boost::boost)
set(SOURCES cppcode/main.cpp)
set(HEADERS cppcode/arrayQueue.hpp cppcode/part.hpp)

add_executable(main ${SOURCES} ${HEADERS})
target_link_libraries(main Boost::program_options)
add_subdirectory(cudacode)
target_include_directories(main PUBLIC cudacode cppcode)
target_link_libraries(main cudaASPL)

add_executable(HillClimb-debug cppcode/HillClimb.cpp ${HEADERS})
target_include_directories(HillClimb-debug PUBLIC cudacode cppcode)
target_link_libraries(HillClimb-debug cudaASPL Boost::program_options)

add_executable(aspl_cuda cppcode/aspl_cuda.cpp cppcode/graph.hpp)
target_include_directories(aspl_cuda PUBLIC cudacode cppcode)
target_link_libraries(aspl_cuda cudaASPLgraph Boost::program_options)

add_executable(beamer cppcode/beamer.cpp ${HEADERS})
target_include_directories(beamer PUBLIC cudacode cppcode)
target_link_libraries(beamer cudaASPL Boost::program_options)
