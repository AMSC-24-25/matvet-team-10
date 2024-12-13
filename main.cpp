#include "cg.hpp"

int main() {

    LinearAlgebra::testCG(); //calling the function testCG from cg.hpp

    return 0;
}

//Compile Command: g++ -std=c++17 -O2 -fopenmp -o cg_solution main.cpp
// Run the Program: ./cg_solutiongit 