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

int
main()
{
    using namespace apsc::LinearAlgebra;
    using namespace Eigen;
    using namespace std;

    using MatrixRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowMatrix = MatrixRow;
    using SpVec = Eigen::VectorXd;
    using ColMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    MPI_Init(nullptr, nullptr);

    int mpi_rank, mpi_size;
    double start_time, end_time, elapsed_time;

    std::vector<double> result;
    std::size_t userDimensionInput;

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    MatrixRow RowDominantMatrix;
    ColMatrix ColumnDominantMatrix;

    SpVec vec;

    std::size_t nRows = 4; // Changed to an odd number
    std::size_t nCols = 4;

    // Creating Matrix & Vector
    RowDominantMatrix = MatrixXd::Random(nRows, nCols);
    ColumnDominantMatrix = MatrixXd::Random(nRows, nCols);

    // Filling Matrix with random values
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            RowDominantMatrix(i, j) = rand() % 10 + 1;
            ColumnDominantMatrix(i, j) = rand() % 10 + 1;
        }
    }
    vec = SpVec::Ones(nRows);

    int rows_per_process = nRows / mpi_size;
    int remainder_rows = nRows % mpi_size; // Handle the odd number of rows
    int local_rows = rows_per_process + (mpi_rank < remainder_rows ? 1 : 0);

    Eigen::MatrixXd A;
    double x[nRows], y[nCols];
    double local_A[local_rows][nCols];
    vector<double> local_y(local_rows, 0.0);


    MPI_Request request_x, request_A, request_y;

    if (mpi_rank == 0)
    {
        cout << "Generated Matrix:" << endl;
        cout << RowDominantMatrix << endl;
    }

    int mpi_result = MPI_Ibcast(vec.data(), nRows, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_x);
    check_mpi_error(mpi_result);

    std::vector<int> send_counts(mpi_size);
    std::vector<int> displacements(mpi_size);

    int offset = 0;
    for (int i = 0; i < mpi_size; i++)
    {
        send_counts[i] = (rows_per_process + (i < remainder_rows ? 1 : 0)) * nCols;
        displacements[i] = offset;
        offset += send_counts[i];
    }

    mpi_result = MPI_Iscatterv(RowDominantMatrix.data(), send_counts.data(), displacements.data(),
                               MPI_DOUBLE, local_A, local_rows * nCols, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_A);
    check_mpi_error(mpi_result);

    mpi_result = MPI_Wait(&request_A, MPI_STATUS_IGNORE);
    check_mpi_error(mpi_result);

    std::cout << "\nProcessor " << mpi_rank + 1 << " received the following rows: \n";

    for (int i = 0; i < local_rows; i++)
    {
        cout << "[";
        for (int j = 0; j < nCols; j++)
        {
            cout << " " << local_A[i][j] << " ";
        }
        cout << "]" << endl;
    }

    mpi_result = MPI_Wait(&request_x, MPI_STATUS_IGNORE);
    check_mpi_error(mpi_result);

    start_time = MPI_Wtime();

    for (int i = 0; i < local_rows; i++)
    {
        local_y[i] = 0.0;
        for (int j = 0; j < nCols; j++)
        {
            local_y[i] += local_A[i][j] * vec[j];
        }
        cout << "\nProcessor " << mpi_rank + 1 << " computed row " << i + 1 << " with a local summed value of: " << local_y[i] << std::endl;
    }

    end_time = MPI_Wtime();

    mpi_result = MPI_Igatherv(local_y.data(), local_rows, MPI_DOUBLE, y, send_counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_y);
    check_mpi_error(mpi_result);

    mpi_result = MPI_Wait(&request_y, MPI_STATUS_IGNORE);
    check_mpi_error(mpi_result);

    if (mpi_rank == 0)
    {
        elapsed_time = end_time - start_time;
        cout << "\nOverall computation time for RowDominant: " << elapsed_time << " seconds" << endl;
        cout << "Result vector y:" << endl;
        cout << "[";
        for (int i = 0; i < nRows; i++)
        {
            cout << y[i] << " ";
        }
        cout << "]" << endl;
    }

    MPI_Finalize();

    return 0;
}
