/*
 * main_matrix.cpp
 *
 *  Created on: 9, Dec 2024
 *      Author: Hirdesh, Mariano, Martina
 */
#include "PMatrix.hpp"
#include "chrono.hpp" // my chrono in Utilities
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
#define MIN_GEN_RANGE 1
#define MAX_GEN_RANGE 5


/*
// Error messages map
std::map<int, std::string> error_messages = {
    {MPI_SUCCESS, "No error"},
    {MPI_ERR_BUFFER, "Invalid buffer pointer"},
    {MPI_ERR_COUNT, "Invalid count argument"},
    {MPI_ERR_TYPE, "Invalid datatype argument"},
    {MPI_ERR_TAG, "Invalid tag argument"},
    {MPI_ERR_COMM, "Invalid communicator"},
    {MPI_ERR_RANK, "Invalid rank"},
    {MPI_ERR_REQUEST, "Invalid request (handle)"},
    {MPI_ERR_ROOT, "Invalid root"},
    {MPI_ERR_GROUP, "Invalid group"},
    {MPI_ERR_OP, "Invalid operation"},
    {MPI_ERR_TOPOLOGY, "Invalid topology"},
    {MPI_ERR_DIMS, "Invalid dimension argument"},
    {MPI_ERR_ARG, "Invalid argument of some other kind"},
    {MPI_ERR_UNKNOWN, "Unknown error"},
    {MPI_ERR_TRUNCATE, "Message truncated on receive"},
    {MPI_ERR_OTHER, "Known error not in this list"},
    {MPI_ERR_INTERN, "Internal MPI (implementation) error"},
    {MPI_ERR_IN_STATUS, "Error code is in status"},
    {MPI_ERR_PENDING, "Pending request"},
    {MPI_ERR_KEYVAL, "Invalid keyval has been passed"},
    {MPI_ERR_NO_MEM, "MPI_ALLOC_MEM failed because memory is exhausted"},
    {MPI_ERR_BASE, "Invalid base passed to MPI_FREE_MEM"},
    {MPI_ERR_INFO_KEY, "Key longer than MPI_MAX_INFO_KEY"},
    {MPI_ERR_INFO_VALUE, "Value longer than MPI_MAX_INFO_VAL"},
    {MPI_ERR_INFO_NOKEY, "Invalid key passed to MPI_INFO_DELETE"},
    {MPI_ERR_SPAWN, "Error in spawning processes"},
    {MPI_ERR_PORT, "Invalid port name passed to MPI_COMM_CONNECT"},
    {MPI_ERR_SERVICE, "Invalid service name passed to MPI_UNPUBLISH_NAME"},
    {MPI_ERR_NAME, "Invalid service name passed to MPI_LOOKUP_NAME"},
    {MPI_ERR_WIN, "Invalid win argument"},
    {MPI_ERR_SIZE, "Invalid size argument"},
    {MPI_ERR_DISP, "Invalid disp argument"},
    {MPI_ERR_INFO, "Invalid info argument"},
    {MPI_ERR_LOCKTYPE, "Invalid locktype argument"},
    {MPI_ERR_ASSERT, "Invalid assert argument"},
    {MPI_ERR_RMA_CONFLICT, "Conflicting accesses to window"},
    {MPI_ERR_RMA_SYNC, "Wrong synchronization of RMA calls"},
    {MPI_ERR_FILE, "Invalid file handle"},
    {MPI_ERR_NOT_SAME, "Collective argument not identical on all processes"},
    {MPI_ERR_AMODE, "Error related to the amode passed to MPI_FILE_OPEN"},
    {MPI_ERR_UNSUPPORTED_DATAREP,
     "Unsupported datarep passed to MPI_FILE_SET_VIEW"},
    {MPI_ERR_UNSUPPORTED_OPERATION, "Unsupported operation, such as seeking on a "
                                     "file which supports sequential access only"},
    {MPI_ERR_NO_SUCH_FILE, "File does not exist"},
    {MPI_ERR_FILE_EXISTS, "File exists"},
    {MPI_ERR_BAD_FILE, "Invalid file name (e.g., path name too long)"},
    {MPI_ERR_ACCESS, "Permission denied"},
    {MPI_ERR_NO_SPACE, "Not enough space"},
    {MPI_ERR_QUOTA, "Quota exceeded"},
    {MPI_ERR_READ_ONLY, "Read-only file or file system"},
    {MPI_ERR_FILE_IN_USE, "File operation could not be completed, as the file is "
                          "currently open by some process"},
    {MPI_ERR_DUP_DATAREP, "Conversion functions could not be registered because "
                          "a data representation identifier was already defined"},
    {MPI_ERR_CONVERSION,
     "An error occurred in a user-supplied data conversion function"},
    {MPI_ERR_IO, "Other I/O error"},
    {MPI_ERR_LASTCODE, "Last error code"}};

// Function to check MPI error codes and print the corresponding error message
void
check_mpi_error(int mpi_result)
{
    if (mpi_result != MPI_SUCCESS)
    {
        std::cout << "MPI Error: " << error_messages[mpi_result] << std::endl;
    }
}*/


int main(int argc, char** argv) {

    using namespace Eigen;
    using namespace std;


    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Matrix and vector dimensions
    const int rows = MAX_ROW;
    const int cols = MAX_COL;

    // Matrix and vector storage
    Eigen::MatrixXd A(rows, cols);
    Eigen::VectorXd x(cols);
    Eigen::VectorXd y(rows);

    // Initialize matrix and vector on root process
     if (mpi_rank == 0) {
        // Seed random number generator
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dist(MIN_GEN_RANGE, MAX_GEN_RANGE); // Random integers between 1 and 10

        // Generate random integer row-major matrix
        Matrix<int, Dynamic, Dynamic, RowMajor> Arowmajor(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                Arowmajor(i, j) = dist(gen); // Populate with random integers
            }
        }

        cout << "Row-Major Random Matrix Arowmajor:\n" << Arowmajor << "\n\n";

        // Print memory layout for row-major
        cout << "Memory layout (row-major):\n";
        cout << "[ ";
        for (int i = 0; i < Arowmajor.size(); ++i) {
            cout << *(Arowmajor.data() + i) << "  ";
        }
        cout << " ]\n\n";

        // Convert row-major matrix to column-major matrix
        Matrix<int, Dynamic, Dynamic, ColMajor> Acolmajor = Arowmajor;
        cout << "Column-Major Matrix Acolmajor:\n" << Acolmajor << "\n\n";

        // Print memory layout for column-major
        cout << "Memory layout (column-major):\n";
        cout << "[ ";

        for (int i = 0; i < Acolmajor.size(); ++i) {
            cout << *(Acolmajor.data() + i) << "  ";
        }
        cout << " ]\n\n";


        // Assign Acolmajor to A as a double matrix for computation
        A = Acolmajor.cast<double>();

        // Initialize vector (ones)
        x = VectorXd::Ones(cols);
    }

    // Broadcast matrix to all processes
    MPI_Bcast(A.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Broadcast vector to all processes
    MPI_Bcast(x.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Distribute work
    int rows_per_process = rows / mpi_size;
    int remainder = rows % mpi_size;

    // Calculate local rows and start row for this process
    int local_rows = rows_per_process + (mpi_rank < remainder ? 1 : 0);
    int start_row = mpi_rank * rows_per_process + (mpi_rank < remainder ? mpi_rank : remainder);

    // Start time measurement
    double start_time = MPI_Wtime();
 
    // Compute local portion of the result
    Eigen::VectorXd local_result = Eigen::VectorXd::Zero(rows);
    for (int i = 0; i < local_rows; ++i) {
        local_result[start_row + i] = A.row(start_row + i).dot(x);

          // Print the row each processor is working on
        cout << "\nProcessor " << mpi_rank + 1 << " received following row: \n";
        cout <<"[ " << A.row(start_row + i) << " ]"<<std::endl;

        // Print computation of each row for the processor
        cout << "\nProcessor " << mpi_rank + 1 << " computes Row " << start_row + i + 1
                  << " with a local summed value of: " << local_result[start_row + i] << std::endl;
    }

    // Gather results
    Eigen::VectorXd global_result(rows);
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
        cout << "Final result vector y: [";
        for (int i = 0; i < rows; ++i) {
            cout << global_result[i] << (i < rows - 1 ? ", " : "");
        }
        cout << "]" << std::endl;
        cout << "\nOverall computation time for RowDominant: " << elapsed_time << " seconds" << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}