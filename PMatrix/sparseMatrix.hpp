#ifndef SPMATRIX_TEST_HPP
#define SPMATRIX_TEST_HPP

#include <iostream>
#include "chrono.hpp"
#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>

using namespace Eigen;
using namespace std;

using SpMat = Eigen::SparseMatrix<double>;
using SpVec = Eigen::VectorXd;

class SPMatrixTest {
private:
    int rows, cols;
    SpMat A;
    SpVec x;

public:
    // Constructor
    SPMatrixTest(int rows = 0, int cols = 0)
        : rows(rows), cols(cols), A(rows, cols), x(cols) {}

    // Initialize matrices and vectors
    void initialize(const string& matrix_file) {
        // Load the matrix from the .mtx file
        if (!Eigen::loadMarket(A, matrix_file)) {
            cerr << "Error loading matrix from file: " << matrix_file << endl;
            exit(1);
        }
        // Initialize vector
        x = SpVec::Ones(A.cols());
    }

    // Run a test
    void run_test() {
        cout << "start run_test with sparse matrix\n";
        // Start timing
        auto start_time = chrono::high_resolution_clock::now();

        // Compute results
        SpVec result = A * x;

        // End timing
        auto end_time = chrono::high_resolution_clock::now();
        auto elapsed_time = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

        // Print results
        cout << "\n--- Sparse Matrix Test ---\n";
        cout << "Final result vector y: [";
        for (int i = 0; i < result.size(); ++i) {
            cout << result[i] << (i < result.size() - 1 ? ", " : "");
        }
        cout << "]" << endl;
        cout << "Computation time: " << elapsed_time << " milliseconds\n";
    }
};
# endif // SPMATRIX_TEST_HPP