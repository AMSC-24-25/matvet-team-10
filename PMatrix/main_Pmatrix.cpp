#include <iostream>
#include <string>
#include <mpi.h>
#include "MeryPmatrix.hpp"
#include "sparsePMatrix.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Check for implementation flag
    if (argc < 2) {
        if (mpi_rank == 0) {
            cerr << "Usage: " << argv[1] << " <implementation_flag> [matrix_file.mtx]" << endl;
            cerr << "Implementation flags: --dense or --sparse" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    string flag = argv[1];

    if (flag == "--dense") {
        cout << "Implementing with Dense matrix\n";
        // Run the dense matrix implementation
        MatrixTest matrix_test;
        matrix_test.initialize(mpi_rank);
        matrix_test.run_tests(mpi_rank, mpi_size);
    } else if (flag == "--sparse") {
        cout << "Implementing with sparse matrix\n";
        // Check for matrix file argument
        if (argc < 3) {
            if (mpi_rank == 0) {
                cerr << "Usage: " << argv << " --sparse <matrix_file.mtx>" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        string matrix_file = argv[2];

        // Run the sparse matrix implementation
        SPMatrixTest matrix_test;
        matrix_test.initialize(mpi_rank, matrix_file);
        matrix_test.run_test(mpi_rank, mpi_size);
    } else {
        if (mpi_rank == 0) {
            cerr << "Invalid implementation flag. Use --dense or --sparse." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}