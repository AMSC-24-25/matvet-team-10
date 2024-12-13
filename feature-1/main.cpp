#include <iostream>
#include <string>
#include "cgg.hpp"

using namespace std;

int main() {
    cout << "Testing Conjugate Gradient algorithm with Dense matrix\n";
    LinearAlgebra::testCG();

    return 0;
}
