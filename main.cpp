#include <iostream>
#include <string>
#include "cg.hpp"

using namespace std;

int main() {
    cout << "Testing Conjugate Gradient algorithm with Dense matrix\n";
    LinearAlgebra::testCG();

    return 0;
}