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
        cout << "Loading matrix from file: " << matrix_file << endl;
        // Load the matrix from the .mtx file
        if (!Eigen::loadMarket(A, matrix_file)) {
            cerr << "Error loading matrix from file: " << matrix_file << endl;
            exit(1);
        }
        rows = A.rows();
        cols = A.cols();
        cout << "Matrix loaded successfully.\n";

        // Initialize vector
        x = SpVec::Ones(cols);
    }

    // Run a test
    void run_test() {
        cout << "Starting matrix-vector multiplication test with matrix size: " << rows << " x " << cols << endl;
        // Start timing
        auto start_time = chrono::high_resolution_clock::now();

        // Compute results
        SpVec b = A * x;

        // End timing
        auto end_time = chrono::high_resolution_clock::now();
        auto elapsed_time = chrono::duration_cast<chrono::duration<double, milli>>(end_time - start_time).count();

        // Save results
        string result_file = "sparse_result.mtx";
        cout << "Saving result in file: " << result_file << endl;
        Eigen::saveMarketVector(b, result_file);

        cout << "Computation time: " << elapsed_time << " milliseconds" << endl;
    }
};

#endif // SPMATRIX_TEST_HPP
