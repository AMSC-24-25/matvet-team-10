#include <iostream>
#include <string>
#include <mpi.h>
#include <omp.h>
#include "MeryPmatrix.hpp"
#include "sparseMatrix.hpp"
#include "cg.hpp"

using namespace std;

int main(int argc, char* argv[]) {

    // Check for implementation flag
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <implementation_flag> [matrix_file.mtx]" << endl;
        cerr << "Implementation flags: --dense or --sparse or --feature1" << endl;
        return 1;
    }

    string flag = argv[1];

    if (flag == "--cg") {
        cout << "Testing Conjugate Gradient algorithm with Dense matrix\n";
        LinearAlgebra::testCG();
    }
    else if (flag == "--dense") {
        // Initialize MPI
        MPI_Init(&argc, &argv);

        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

        // Run the dense matrix implementation
        MatrixTest matrix_test;
        matrix_test.initialize(mpi_rank);
        matrix_test.run_tests(mpi_rank, mpi_size);

        // Finalize MPI
        MPI_Finalize();
    } else if (flag == "--sparse") {
        cout << "Implementing with sparse matrix in scalar mode\n";
        // Check for matrix file argument
        if (argc < 3) {
            cerr << "Usage: " << argv[0] << " --sparse <matrix_file.mtx>" << endl;
            return 1;
        }

        string matrix_file = argv[2];

        // Run the sparse matrix implementation
        SPMatrixTest matrix_test;
        matrix_test.initialize(matrix_file);
        matrix_test.run_test();
    } 
   else {
        cerr << "Invalid implementation flag. Use --dense-parallel, --sparse-scalar, or --cg-test." << endl;
        return 1;
    }
    return 0;
}