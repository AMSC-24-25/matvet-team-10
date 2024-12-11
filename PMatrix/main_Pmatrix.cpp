/*
 * main_matrix.cpp
 *
 *  Created on: 9, Dec 2024
 *      Author: Hirdesh, Mariano, Martina
*/

#include "PMatrix.hpp"
#include "chrono.hpp" 
#include <iostream>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include </u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3/Eigen/Dense>
#include </u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3/Eigen/Sparse>
#include </u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3/unsupported/Eigen/SparseExtra>
#include <mpi.h>
#include <cmath>
#pragma GCC diagnostic pop

#define MAX_ROW 5
#define MAX_COL 5
#define MIN_GENERATION_RANGE 1
#define MAX_GENERATION_RANGE 5
#define PRINT_COMPUTATION_LOG false

using namespace Eigen;
using namespace std;

// Function to compute and test matrix-vector multiplication
void run_test(const MatrixXd& A, const VectorXd& x, int mpi_rank, int mpi_size, const string& test_name) {
    int rows = A.rows();
    int cols = A.cols();

    // Broadcast matrix to all processes
    MPI_Bcast(const_cast<double*>(A.data()), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Broadcast vector to all processes
    MPI_Bcast(const_cast<double*>(x.data()), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Distribute work
    int rows_per_process = rows / mpi_size;
    int remainder = rows % mpi_size;

    // Calculate local rows and start row for this process
    int local_rows = rows_per_process + (mpi_rank < remainder ? 1 : 0);
    int start_row = mpi_rank * rows_per_process + (mpi_rank < remainder ? mpi_rank : remainder);

    // Start time measurement
    double start_time = MPI_Wtime();

    // Compute local portion of the result
    VectorXd local_result = VectorXd::Zero(rows);
    for (int i = 0; i < local_rows; ++i) {
        local_result[start_row + i] = A.row(start_row + i).dot(x);

        if (PRINT_COMPUTATION_LOG) {
            // Print the row each processor is working on
            cout << "\nProcessor " << mpi_rank + 1 << " received following row: \n";
            cout << "[ " << A.row(start_row + i) << " ]" << endl;

            // Print computation of each row for the processor
            cout << "\nProcessor " << mpi_rank + 1 << " computes Row " << start_row + i + 1
                 << " with a local summed value of: " << local_result[start_row + i] << endl;
        }
    }

    // Gather results
    VectorXd global_result(rows);
    MPI_Reduce(
        local_result.data(), 
        global_result.data(), 
        rows, 
        MPI_DOUBLE, 
        MPI_SUM, 
        0, 
        MPI_COMM_WORLD
    );

    // End time measurement
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    // Print results on root process
    if (mpi_rank == 0) {
        cout << "\n--- " << test_name << " ---\n";
        cout << "Final result vector y: [";
        for (int i = 0; i < rows; ++i) {
            cout << global_result[i] << (i < rows - 1 ? ", " : "");
        }
        cout << "]" << endl;
        cout << "Overall computation time: " << elapsed_time << " seconds" << endl;
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Matrix and vector dimensions
    const int rows = MAX_ROW;
    const int cols = MAX_COL;

    // Matrix and vector storage
    MatrixXd Arow(rows, cols);
    MatrixXd Acol(rows, cols);
    VectorXd x(cols);

    // Initialize matrix and vector on root process
    if (mpi_rank == 0) {
        // Seed random number generator
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dist(MIN_GENERATION_RANGE, MAX_GENERATION_RANGE);

        // Generate random integer row-major matrix
        Matrix<int, Dynamic, Dynamic, RowMajor> Arowmajor(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                Arowmajor(i, j) = dist(gen);
            }
        }

        // Convert row-major matrix to column-major matrix
        Matrix<int, Dynamic, Dynamic, ColMajor> Acolmajor = Arowmajor;

        // Cast to double for computation
        Arow = Arowmajor.cast<double>();
        Acol = Acolmajor.cast<double>();

        // Initialize vector (ones)
        x = VectorXd::Ones(cols);
    }

    // Run tests for row-major and column-major matrices
    run_test(Arow, x, mpi_rank, mpi_size, "Row-Major Test");
    run_test(Acol, x, mpi_rank, mpi_size, "Column-Major Test");

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
