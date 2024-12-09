/*
 * main_matrix.cpp
 *
 *  Created on: 9, Dec 2024
 *      Author: Hirdesh, Mariano, Martina
 */
#include <iostream>
#include "PMatrix.hpp"
#include "chrono.hpp" // my chrono in Utilities
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#include <mpi.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
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
    {MPI_ERR_UNSUPPORTED_DATAREP, "Unsupported datarep passed to MPI_FILE_SET_VIEW"},
    {MPI_ERR_UNSUPPORTED_OPERATION, "Unsupported operation, such as seeking on a file which supports sequential access only"},
    {MPI_ERR_NO_SUCH_FILE, "File does not exist"},
    {MPI_ERR_FILE_EXISTS, "File exists"},
    {MPI_ERR_BAD_FILE, "Invalid file name (e.g., path name too long)"},
    {MPI_ERR_ACCESS, "Permission denied"},
    {MPI_ERR_NO_SPACE, "Not enough space"},
    {MPI_ERR_QUOTA, "Quota exceeded"},
    {MPI_ERR_READ_ONLY, "Read-only file or file system"},
    {MPI_ERR_FILE_IN_USE, "File operation could not be completed, as the file is currently open by some process"},
    {MPI_ERR_DUP_DATAREP, "Conversion functions could not be registered because a data representation identifier was already defined"},
    {MPI_ERR_CONVERSION, "An error occurred in a user-supplied data conversion function"},
    {MPI_ERR_IO, "Other I/O error"},
    {MPI_ERR_LASTCODE, "Last error code"}
};

// Function to check MPI error codes and print the corresponding error message
void check_mpi_error(int mpi_result) {
    if (mpi_result != MPI_SUCCESS) {
        std::cout << "MPI Error: " << error_messages[mpi_result] << std::endl;
    }
}

int main() {
    using namespace apsc::LinearAlgebra;
    using namespace Eigen;
    using namespace std;

    using MatrixRow = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using RowMatrix = MatrixRow;
    using SpVec = Eigen::VectorXd;
    using ColMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

    MPI_Init(nullptr, nullptr);

    int mpi_rank;
    int mpi_size;
    double time_scalar;
    double time_parallel;
    double time_setup;
    std::vector<double> result;

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    MatrixRow mr;
    SpVec vec;

    std::size_t nRows = 4;
    std::size_t nCols = 4;

    // Creating Matrix & Vector
    mr = MatrixXd::Random(nRows, nCols);
    
    // Filling Matrx with random values postive integers
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            mr(i, j) = rand() % 10 + 1;
        }
    }
    vec = SpVec::Ones(nRows);

  
   //std::cout << "Allocating vector of size: " << vec.size() << std::endl;
   // Vector dimmension
   //std::cout << "Vector size: "<<vec.size()<<std::endl;
   //std::cout << "Vector data: "<<vec<<std::endl;
    int rows_per_process = nRows / mpi_size;
    Eigen::MatrixXd A;
    double x[nRows], y[nCols];
    double local_A[rows_per_process][nRows], local_y[rows_per_process];

    MPI_Request request_x, request_A, request_y;

    if (mpi_rank == 0) {
        cout << mr << endl;
    }

//cout << "fine line 131" << endl;
    // Step 1: Non-blocking broadcast vector x
    int mpi_result = MPI_Ibcast(vec.data(), mpi_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_x);
    check_mpi_error(mpi_result);

//cout << "fine line 136" << endl;
    // Step 2: Non-blocking scatter rows of matrix A
mpi_result = MPI_Iscatter(mr.data(), rows_per_process * nCols, MPI_DOUBLE, local_A, rows_per_process * nCols, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_A);
check_mpi_error(mpi_result);

// Wait for the scatter to complete
mpi_result = MPI_Wait(&request_A, MPI_STATUS_IGNORE); // Wait for rows of A to be scattered
check_mpi_error(mpi_result);

// Print local_A in each process to debug
std::cout << "Process " << mpi_rank << " received local_A:\n";
for (int i = 0; i < rows_per_process; i++) {
    for (int j = 0; j < nCols; j++) {
        std::cout << local_A[i][j] << " ";
    }
    std::cout << std::endl;
}



    // Step 3: Wait for data to arrive and perform computation
    mpi_result = MPI_Wait(&request_x, MPI_STATUS_IGNORE); // Wait for x to be broadcasted
    check_mpi_error(mpi_result);
    mpi_result = MPI_Wait(&request_A, MPI_STATUS_IGNORE); // Wait for rows of A to be scattered
    check_mpi_error(mpi_result);


//cout << "fine line 148" << endl;

      for (int i = 0; i < rows_per_process; i++) {
        local_y[i] = 0.0;
        for (int j = 0; j < nCols; j++) {
          
            local_y[i] += local_A[i][j] * vec[j];
    }
         std::cout << "After summing, processor   = " << local_y[i] << std::endl;
}
//cout << "fine line 156" << endl;
    // Step 4: Non-blocking gather partial results
    mpi_result = MPI_Igather(local_y, rows_per_process, MPI_DOUBLE, y,
                              rows_per_process, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request_y);
    check_mpi_error(mpi_result);
//cout << "fine line 160" << endl;
    // Wait for the gather to complete
    mpi_result = MPI_Wait(&request_y, MPI_STATUS_IGNORE);
    check_mpi_error(mpi_result);
//cout << "fine line 165" << endl;
    // Print the result vector on the root process
    if (mpi_rank == 0) {
        printf("Result vector y:\n");
        for (int i = 0; i < nCols; i++) {
            printf("%f\n", y[i]);
        }
    }
//cout << "fine line 173" << endl;

 


/*





  apsc::PMatrix<RowMatrix> pmr;
  pmr.setup(mr,mpi_comm);
  int rank=0;
  while (rank < mpi_size)
    {
      if(mpi_rank==rank)
        {
          std::cout<<"Process rank="<< mpi_rank<<" Local Matrix=";
          std::cout<<pmr.getLocalMatrix();
       }
      rank++;
      MPI_Barrier(mpi_comm);
    }

  ColMatrix mc;
  if(mpi_rank==0)
     { std::cout<<"Creating a 4 by 4 Matrix of double, col-major\n";
     mc.resize(nRows,nCols);
     for (auto i=0u;i<mc.rows();++i)
       for (auto j=0u;j<mc.cols();++j)
         mc(i,j)=i+j;
     }

  apsc::PMatrix<ColMatrix> pmc;
  pmc.setup(mc,mpi_comm);
  rank=0;
  while (rank < mpi_size)
    {
      if(mpi_rank==rank)
        {
          std::cout<<"Process rank="<< mpi_rank<<" Local Matrix=";
          std::cout<<pmc.getLocalMatrix();
        }
      rank++;
      MPI_Barrier(mpi_comm);
    }

  Eigen::VectorXd ones = Eigen::VectorXd::Ones(nCols);
  std::vector<double> ones_vec(ones.data(), ones.data() + ones.size());
  pmr.product(ones_vec);
  // Only for debugging
// you need to make data public in PMatrix!
//  rank=0;
//  while (rank < mpi_size)
//    {
//      if(mpi_rank==rank)
//        {
//          std::cout<<"Process rank="<< mpi_rank<<" Local Product=";
//          for (auto i=0u;i<pmr.localProduct.size();++i)
//            std::cout<<pmr.localProduct[i]<<", ";
//          std::cout<<std::endl;
//        }
//      rank++;
//      MPI_Barrier(mpi_comm);
//    }


  std::vector<double> result;
  pmr.collectGlobal(result);
  if(mpi_rank==0)
    {
      std::cout<<"Testing row major version and all-to-one communication, residual should be 0\n";
      Eigen::VectorXd exact = mr * ones;
      double residual=0.0;
      for (auto i=0u;i<exact.size();++i)
        residual+=(exact[i]-result[i])*(exact[i]-result[i]);
      std::cout<<"Residual="<<residual<<std::endl;
    }


  // Now column major
  std::vector<double> ones_vec(ones.data(), ones.data() + ones.size());
  pmc.product(ones_vec);
//   rank=0;
// Only for debugging
// you need to make data public in PMatrix!
//   while (rank < mpi_size)
//     {
//       if(mpi_rank==rank)
//         {
//           std::cout<<"Process rank="<< mpi_rank<<" Local Product=";
//           for (auto i=0u;i<pmc.localProduct.size();++i)
//             std::cout<<pmc.localProduct[i]<<", ";
//           std::cout<<std::endl;
//         }
//       rank++;
//       MPI_Barrier(mpi_comm);
//     }


   pmc.AllCollectGlobal(result);
   if(mpi_rank==mpi_size-1)
     {
       std::cout<<"Testing column major version and all-to-all communication, residual should be 0\n";
       auto exact=mr*ones;
       double residual=0.0;
       for (auto i=0u;i<exact.size();++i)
         residual+=(exact[i]-result[i])*(exact[i]-result[i]);
       std::cout<<"Residual="<<residual<<std::endl;
     }
   MPI_Barrier(mpi_comm);

   if(mpi_rank==0)
     std::cout<<"\nNow a more serious test"<<std::endl;
   Timings::Chrono clock;
   constexpr std::size_t N=1500;
   Eigen::MatrixXd A;
   std::vector<double> uno(N,1.0);
   double time_scalar;
   double time_parallel;
   double time_setup;
   if(mpi_rank==0)
     {
       A.resize(N,N);
       A.setRandom();// a random NxN matrix
       clock.start();
       Eigen::VectorXd uno_vec = Eigen::Map<Eigen::VectorXd>(uno.data(), uno.size());
       Eigen::VectorXd temp_result = A * uno_vec;
       result.assign(temp_result.data(), temp_result.data() + temp_result.size());
       clock.stop();
       time_scalar=clock.wallTime();
       clock.start();
     }
   MPI_Barrier(mpi_comm);
   apsc::PMatrix<RowMatrix> Ap;
   Ap.setup(A, mpi_comm);
   if(mpi_rank==0)
     {
       clock.stop();
       time_setup=clock.wallTime();
       clock.start();
     }
   MPI_Barrier(mpi_comm);
   Ap.product(uno);
   Ap.collectGlobal(result);
   if(mpi_rank==0)
     {
       clock.stop();
       time_parallel=clock.wallTime();
       std::cout<<"Scalar time="<<time_scalar<<" Setup time="<<time_setup<<" Parallel product time="<<time_parallel<<std::endl;
       std::cout<<"Including setup: Speed-up="<<time_scalar/(time_setup+time_parallel)<<
           " Efficiency="<<(1./mpi_size)*time_scalar/(time_setup+time_parallel)<<std::endl;
       std::cout<<"Excluding setup: Speed-up="<<time_scalar/time_parallel<<
            " Efficiency="<<(1./mpi_size)*time_scalar/time_parallel<<std::endl;
     }



  //

*/

  MPI_Finalize();

return 0;

}