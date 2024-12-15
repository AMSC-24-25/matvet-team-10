#include "MeryPmatrix.hpp"
#include <mpi.h>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Create MatrixTest instance
    MatrixTest matrix_test;

    // Initialize matrices and vectors
    matrix_test.initialize(mpi_rank);

    // Run tests
    matrix_test.run_tests(mpi_rank, mpi_size);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
