#ifndef SPMATRIX_TEST_HPP
#define SPMATRIX_TEST_HPP

#include <iostream>
#include <mpi.h>
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
    void initialize(int mpi_rank, const string& matrix_file) {
        cout << "Start initialize for sparse matrix\n";
        if (mpi_rank == 0) {
            // Load the matrix from the .mtx file
            if (!Eigen::loadMarket(A, matrix_file)) {
                cerr << "Error loading matrix from file: " << matrix_file << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            cout << "Loaded matrix file successfully\n";
            // Initialize vector
            x = SpVec::Ones(A.cols());
        }

        // Broadcast matrix dimensions
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Resize matrix and vector for all processes
        if (mpi_rank != 0) {
            A.resize(rows, cols);
            x.resize(cols);
        }

        // Broadcast matrix and vector
        MPI_Bcast(const_cast<double*>(A.valuePtr()), A.nonZeros(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<int*>(A.outerIndexPtr()), A.outerSize(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<int*>(A.innerIndexPtr()), A.nonZeros(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<double*>(x.data()), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        cout << "Exiting Initialize\n";
    }

    // Run a test
    void run_test(int mpi_rank, int mpi_size) {
        cout << "start run_test with sparse matrix\n";
        // Distribute work
        int rows_per_process = rows / mpi_size;
        int remainder = rows % mpi_size;

        int local_rows = rows_per_process + (mpi_rank < remainder ? 1 : 0);
        int start_row = mpi_rank * rows_per_process + (mpi_rank < remainder ? mpi_rank : remainder);

        // Start timing
        double start_time = MPI_Wtime();

        // Compute local results
        cout << "SPMatrix: Started MPI\n";
        SpVec local_result = SpVec::Zero(rows);
        for (int i = 0; i < local_rows; ++i) {
            local_result[start_row + i] = A.row(start_row + i).dot(x);
        }
        cout << "SPMatrix: Computed local results\n";
        // Gather results
        SpVec global_result(rows);
        MPI_Reduce(local_result.data(), global_result.data(), rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        cout << "SPMatrix: Gather global results\n";
        // End timing
        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;

        // Print results
        if (mpi_rank == 0) {
            cout << "\n--- Sparse Matrix Test ---\n";
            cout << "Final result vector y: [";
            for (int i = 0; i < rows; ++i) {
                cout << global_result[i] << (i < rows - 1 ? ", " : "");
            }
            cout << "]" << endl;
            cout << "Computation time: " << elapsed_time << " seconds\n";
        }
    }
};

#endif // SPMATRIX_TEST_HPP