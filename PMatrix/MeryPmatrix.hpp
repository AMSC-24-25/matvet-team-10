#ifndef MATRIX_TEST_HPP
#define MATRIX_TEST_HPP

#include <iostream>
#include <random>
#include <mpi.h>
#include </u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3/Eigen/Dense>

#define MAX_ROW 5
#define MAX_COL 5
#define MIN_GENERATION_RANGE 1
#define MAX_GENERATION_RANGE 5
#define PRINT_COMPUTATION_LOG false

using namespace Eigen;
using namespace std;

class MatrixTest {
private:
    int rows, cols;
    MatrixXd Arow, Acol;
    VectorXd x;

public:
    // Constructor
    MatrixTest(int rows = MAX_ROW, int cols = MAX_COL)
        : rows(rows), cols(cols), Arow(rows, cols), Acol(rows, cols), x(cols) {}

    // Initialize matrices and vectors
    void initialize(int mpi_rank) {
        if (mpi_rank == 0) {
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

            // Convert row-major to column-major
            Matrix<int, Dynamic, Dynamic, ColMajor> Acolmajor = Arowmajor;

            // Cast to double
            Arow = Arowmajor.cast<double>();
            Acol = Acolmajor.cast<double>();

            // Initialize vector
            x = VectorXd::Ones(cols);
        }
    }

    // Run a test
    void run_test(const MatrixXd& A, int mpi_rank, int mpi_size, const string& test_name) {
        // Broadcast matrix and vector
        MPI_Bcast(const_cast<double*>(A.data()), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<double*>(x.data()), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Distribute work
        int rows_per_process = rows / mpi_size;
        int remainder = rows % mpi_size;

        int local_rows = rows_per_process + (mpi_rank < remainder ? 1 : 0);
        int start_row = mpi_rank * rows_per_process + (mpi_rank < remainder ? mpi_rank : remainder);

        // Start timing
        double start_time = MPI_Wtime();

        // Compute local results
        VectorXd local_result = VectorXd::Zero(rows);
        for (int i = 0; i < local_rows; ++i) {
            local_result[start_row + i] = A.row(start_row + i).dot(x);

            if (PRINT_COMPUTATION_LOG) {
                cout << "\nProcessor " << mpi_rank + 1 << " received row: ["
                     << A.row(start_row + i) << "]\n";
                cout << "Processor " << mpi_rank + 1 << " computes Row " << start_row + i + 1
                     << " with sum: " << local_result[start_row + i] << endl;
            }
        }

        // Gather results
        VectorXd global_result(rows);
        MPI_Reduce(local_result.data(), global_result.data(), rows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // End timing
        double end_time = MPI_Wtime();
        double elapsed_time = end_time - start_time;

        // Print results
        if (mpi_rank == 0) {
            cout << "\n--- " << test_name << " ---\n";
            cout << "Final result vector y: [";
            for (int i = 0; i < rows; ++i) {
                cout << global_result[i] << (i < rows - 1 ? ", " : "");
            }
            cout << "]" << endl;
            cout << "Computation time: " << elapsed_time << " seconds\n";
        }
    }

    // Run both tests
    void run_tests(int mpi_rank, int mpi_size) {
        run_test(Arow, mpi_rank, mpi_size, "Row-Major Test");
        run_test(Acol, mpi_rank, mpi_size, "Column-Major Test");
    }
};

#endif // MATRIX_TEST_HPP
