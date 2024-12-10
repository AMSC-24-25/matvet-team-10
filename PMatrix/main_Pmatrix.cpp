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
}


int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Matrix and vector dimensions
    const int rows = 5;
    const int cols = 5;

    // Matrix and vector storage
    Eigen::MatrixXd A(rows, cols);
    Eigen::VectorXd x(cols);
    Eigen::VectorXd y(rows);

    // Initialize matrix and vector on root process
    if (mpi_rank == 0) {
        // Define matrix
        A << 1, 2, 3, 4,10,
             5, 6, 7, 8,20,
             9, 10, 11, 12,30,
             13, 14, 15, 16,40,
              17, 18, 3, 4,50;


        // Define vector (ones)
        x = Eigen::VectorXd::Ones(cols);
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
    int start_row = mpi_rank * rows_per_process + 
                    (mpi_rank < remainder ? mpi_rank : remainder);

    // Compute local portion of the result
    Eigen::VectorXd local_result = Eigen::VectorXd::Zero(rows);
    for (int i = 0; i < local_rows; ++i) {
        local_result[start_row + i] = A.row(start_row + i).dot(x);
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

    // Print results on root process
    if (mpi_rank == 0) {
        std::cout << "Final result vector y: [";
        for (int i = 0; i < rows; ++i) {
            std::cout << global_result[i] << (i < rows - 1 ? ", " : "");
        }
        std::cout << "]" << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}