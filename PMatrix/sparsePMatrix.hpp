#ifndef SPMATRIX_TEST_HPP
#define SPMATRIX_TEST_HPP

#include <iostream>
#include <random>
#include <mpi.h>
#include <Eigen/SparseCore>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>


#define MAX_ROW 5
#define MAX_COL 5
#define MIN_GENERATION_RANGE 1
#define MAX_GENERATION_RANGE 5
#define PRINT_COMPUTATION_LOG false

using namespace Eigen;
using namespace std;

using SpMat=Eigen::SparseMatrix<double>;
using SpVec=Eigen::VectorXd;

class SPMatrixTest {
private:
    int rows, cols;
    SpMat A1, Arow, Acol;
    SpVec x;

public:
    // Constructor
    SPMatrixTest(int rows = MAX_ROW, int cols = MAX_COL)
        : rows(rows), cols(cols), Arow(rows, cols), Acol(rows, cols), x(cols) {}

    // Initialize matrices and vectors
    void initialize(int mpi_rank) {
        if (mpi_rank == 0) {
            random_device rd;
            mt19937 gen(rd());
            uniform_int_distribution<> dist(MIN_GENERATION_RANGE, MAX_GENERATION_RANGE);

            // // Generate random sparse matrix
            // vector<Triplet<double>> tripletList;
            // for (int i = 0; i < rows; ++i) {
            //     for (int j = 0; j < cols; ++j) {
            //         if (dist(gen) % 2 == 0) { // Randomly decide if the element is non-zero
            //             tripletList.push_back(Triplet<double>(i, j, dist(gen)));
            //         }
            //     }
            // }

            // // Create sparse matrices
            // Arow.setFromTriplets(tripletList.begin(), tripletList.end());
            // Acol = Arow;

            //load the matrix file that will be downloaded from matrix market
            Eigen::loadMarket(A1, "spd_matrix.mtx");

            // Initialize vector
            x = SpVec::Ones(cols);
        }
    }

    // Run a test
    void run_test(const SpMat& A, int mpi_rank, int mpi_size, const string& test_name) {
        // Broadcast matrix and vector
        MPI_Bcast(const_cast<double*>(A.valuePtr()), A.nonZeros(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<int*>(A.outerIndexPtr()), A.outerSize(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<int*>(A.innerIndexPtr()), A.nonZeros(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(const_cast<double*>(x.data()), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Distribute work
        int rows_per_process = rows / mpi_size;
        int remainder = rows % mpi_size;

        int local_rows = rows_per_process + (mpi_rank < remainder ? 1 : 0);
        int start_row = mpi_rank * rows_per_process + (mpi_rank < remainder ? mpi_rank : remainder);

        // Start timing
        double start_time = MPI_Wtime();

        // Compute local results
        SpVec local_result = SpVec::Zero(rows);
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
        SpVec global_result(rows);
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
        run_test(Arow, mpi_rank, mpi_size, "Row-Major Test"); //what to write for matrix loaded?
        run_test(Acol, mpi_rank, mpi_size, "Column-Major Test"); //what to write for matrix loaded?
    }
};

#endif // SPMATRIX_TEST_HPP