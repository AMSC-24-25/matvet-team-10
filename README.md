# TASK 1: CG Implementation with OpenMP

The **Conjugate Gradient (CG)** function is the core component of this implementation. It solves the linear system \( Ax = b \) step by step, where \( A \) is a symmetric positive definite (SPD) matrix.

#### Aim

To solve the equation \( Ax = b \), where:
- **A**: Coefficient matrix (SPD).
- **x**: Initial guess for the solution (modified iteratively).
- **b**: Right-hand side vector.

The CG function aims to compute the solution iteratively, stopping when the desired tolerance is reached or when the maximum number of iterations is exceeded.

---

### **Inputs and Outputs**

**Inputs:**
- `A`: The matrix of the system.
- `x`: An initial guess for the solution.
- `b`: The right-hand side vector.
- `max_iter`: The maximum number of iterations before stopping.
- `tol`: The acceptable margin of error.

**Outputs:**
- Returns `0` if a solution is found within the stipulated steps and `1` otherwise.
- Resets `x` to the closest approximation of the solution.
- Resets `tol` to the final error.

---

### **How It Works**

1. **Residual Calculation**:
   - Calculate the starting residual: \( r = b - Ax \).
   - Check if the initial guess for `x` is acceptable by comparing the residual against the tolerance.

2. **Iterative Refinement**:
   - If the residual is too large:
     1. Compute the step size (**alpha-factor**) and update the direction vector (**beta-factor**).
     2. Update the residual, working direction vector \( p \), and the solution \( x \).

3. **Stopping Conditions**:
   - Stop when the residual is within the tolerance or when the maximum number of iterations is reached.

---

### **Parallelization**

This implementation uses **OpenMP** to parallelize operations, accelerating:
- Residual calculations.
- Vector updates.
- Matrix-vector multiplications.

---

### **Dense Matrix Class**

The **DenseMatrix** class provides:
- Support for matrix storage in **row-major** and **column-major** order.
- Random SPD matrix generation using `randomFillSPD`.
- Optimized matrix-vector multiplication using OpenMP.

**Key Features**:
- Handles SPD matrices required for the CG method.
- Provides methods for matrix and vector multiplication (e.g., q = A * p ).

---

### **Test Suite**

The `runTest` function evaluates the CG implementation:
1. A DenseMatrix is created and filled with random SPD values.
2. The CG function is invoked to solve the system using the DenseMatrix for operations.
3. Results are recorded for matrices of various sizes and storage orders (row-major and column-major).
4. Performance is analyzed by measuring convergence and computation time.

---

### **How to Use It**
**inside PMatrix**

1. **Compile the code**:
   ```bash
    make
   ```

2. **Run the CG tests**:
   ```bash
     ./main_Pmatrix --cg
   ```

---

### **Example Output**

Here’s an example of the output:

```
Testing Conjugate Gradient algorithm with Dense matrix

-------Testing with small SPD matrices: size = 5. ---------------

Matrix storage order: Row-major
Thread 7 of 16 is working.
Thread 14 of 16 is working.
Thread 10 of 16 is working.
Thread 8 of 16 is working.
Thread 11 of 16 is working.
Thread 5 of 16 is working.
Thread 9 of 16 is working.
Thread 3 of 16 is working.
Thread 2 of 16 is working.
Thread 1 of 16 is working.
Thread 6 of 16 is working.
Thread 15 of 16 is working.
Thread 4 of 16 is working.
Thread 0 of 16 is working.
Thread 12 of 16 is working.
Thread 13 of 16 is working.
Result: Converged, Iterations: 5, Residual: 1.85233e-18
Computation time: 19.7712 ms

Matrix storage order: Column-major
Thread 11 of 16 is working.
Thread 6 of 16 is working.
Thread 0 of 16 is working.
Thread 8 of 16 is working.
Thread 4 of 16 is working.
Thread 2 of 16 is working.
Thread 1 of 16 is working.
Thread 7 of 16 is working.
Thread 13 of 16 is working.
Thread 15 of 16 is working.
Thread 10 of 16 is working.
Thread 12 of 16 is working.
Thread 3 of 16 is working.
Thread 9 of 16 is working.
Thread 14 of 16 is working.
Thread 5 of 16 is working.
Result: Converged, Iterations: 5, Residual: 7.01458e-18
Computation time: 0.530729 ms

-------Testing with medium SPD matrices: size = 50. ---------------

Matrix storage order: Row-major
Thread 13 of 16 is working.
Thread 12 of 16 is working.
Thread 6 of 16 is working.
Thread 2 of 16 is working.
Thread 4 of 16 is working.
Thread 11 of 16 is working.
Thread 8 of 16 is working.
Thread 7 of 16 is working.
Thread 5 of 16 is working.
Thread 15 of 16 is working.
Thread 10 of 16 is working.
Thread 14 of 16 is working.
Thread 9 of 16 is working.
Thread 0 of 16 is working.
Thread 1 of 16 is working.
Thread 3 of 16 is working.
Result: Converged, Iterations: 5, Residual: 8.02327e-08
Computation time: 6.76917 ms

Matrix storage order: Column-major
Thread 8 of 16 is working.
Thread 4 of 16 is working.
Thread 5 of 16 is working.
Thread 7 of 16 is working.
Thread 3 of 16 is working.
Thread 14 of 16 is working.
Thread 2 of 16 is working.
Thread 12 of 16 is working.
Thread 6 of 16 is working.
Thread 13 of 16 is working.
Thread 11 of 16 is working.
Thread 10 of 16 is working.
Thread 9 of 16 is working.
Thread 15 of 16 is working.
Thread 1 of 16 is working.
Thread 0 of 16 is working.
Result: Converged, Iterations: 5, Residual: 4.74391e-08
Computation time: 0.38678 ms

Results documented for analysis.
```

---

### **Comments on the Results**

1. **Convergence**:
   - The CG algorithm successfully converged for all test cases within 5 iterations, demonstrating stability and effectiveness.

2. **Residuals**:
   - The residuals are extremely small (e.g., `1.85233e-18`), confirming the accuracy of the solution.

3. **Computation Time**:
   - **Column-major storage** performed significantly faster than row-major storage, highlighting the impact of memory layout on performance.
   - For example, with medium SPD matrices:
     - Row-major: `6.76917 ms`
     - Column-major: `0.38678 ms`

# TASK 2 Matrix-Vector Multiplication with MPI

This implementation is part of **features2** in the **matvet-team-10** project. It focuses on improving matrix-vector multiplication by using **MPI (Message Passing Interface)** to parallelize the workload. By replacing the `Matrix` class with the `PMatrix` class, we can run the computation in a distributed environment, making it faster and more scalable.

#### What It Does

1. **Initialization**:
   - A random matrix (`A`) and a vector (`x`) are created on the main process (rank 0).
   - The matrix and vector are then shared with all the other processes using `MPI_Bcast`.

2. **Parallel Computation**:
   - The rows of the matrix are divided among the available processes.
   - Each process works on its chunk of rows, calculating the dot products with the vector.

3. **Gathering Results**:
   - Once all processes finish their part, the results are combined on the main process using `MPI_Reduce`.

4. **Testing**:
   - The program runs tests with both row-major and column-major layouts of the matrix to ensure everything works correctly.
   - It also measures how long each test takes.

#### Why Use MPI?

MPI is a great choice for operations like this because:
- It **splits the workload** across multiple processes, so the computation happens faster.
- It’s **scalable**, meaning it can handle much larger matrices by using multiple machines.
- MPI is built for **high-performance environments**, so it minimizes delays when processes communicate.

### How to Use It

1. **Compile the program**:
   ```bash
   root@9f599bdc117f PMatrix # make
    mpic++ -std=c++17  -O2 -fopenmp -I.  -I /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3   -c main_Pmatrix.cpp -o main_Pmatrix.o
   mpic++ -std=c++17  -O2 -fopenmp -I.  -I /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3   main_Pmatrix.o -o main_Pmatrix
   ```

2. **Run the program**:
   - By default:
     ```bash
     root@9f599bdc117f PMatrix # ./main_Pmatrix --dense
     ```
   - To specify the number of MPI processes (threads):
     ```bash
     root@9f599bdc117f PMatrix # mpirun -n <number_of_processes> ./main_Pmatrix --dense
     ```
   For example:
   ```bash
   root@9f599bdc117f PMatrix # mpirun -n 4 ./main_Pmatrix --dense
   ```
#### Example Output

Here’s an example of what the program might print when you run it:

```
--- Row-Major Test ---
Final result vector y: [290, 321, 308, 283, 271, 284, 323, 303, 290, 307, 293, 300, 278, 317, 317, 282, 294, 304, 301, 280, 305, 317, 299, 293, 298, 318, 332, 329, 314, 298, 297, 300, 286, 299, 314, 301, 289, 285, 316, 299, 299, 294, 293, 295, 284, 279, 279, 297, 295, 296, 314, 281, 272, 315, 291, 301, 301, 304, 300, 322, 307, 297, 307, 277, 303, 312, 303, 299, 282, 292, 322, 313, 309, 310, 278, 329, 282, 308, 299, 291, 303, 290, 299, 305, 313, 294, 284, 304, 294, 292, 308, 303, 295, 311, 297, 304, 315, 332, 316, 297]
Computation time: 7.851e-06 seconds

--- Column-Major Test ---
Final result vector y: [290, 321, 308, 283, 271, 284, 323, 303, 290, 307, 293, 300, 278, 317, 317, 282, 294, 304, 301, 280, 305, 317, 299, 293, 298, 318, 332, 329, 314, 298, 297, 300, 286, 299, 314, 301, 289, 285, 316, 299, 299, 294, 293, 295, 284, 279, 279, 297, 295, 296, 314, 281, 272, 315, 291, 301, 301, 304, 300, 322, 307, 297, 307, 277, 303, 312, 303, 299, 282, 292, 322, 313, 309, 310, 278, 329, 282, 308, 299, 291, 303, 290, 299, 305, 313, 294, 284, 304, 294, 292, 308, 303, 295, 311, 297, 304, 315, 332, 316, 297]
Computation time: 5.579e-06 seconds
```

#### Commenting the Results

1. **Final Result Vector**:
   - Both the row-major and column-major tests produce the **same final result vector** (`y`), confirming that the implementation is correct and independent of the matrix layout.
   - The result vector represents the dot products of each row in the matrix with the vector `x`.

2. **Computation Time**:
   - The **row-major test** took slightly longer (`7.851e-06 seconds`) than the **column-major test** (`5.579e-06 seconds`).
   - This difference could be attributed to how the data is accessed in memory: column-major access may have benefited from better cache utilization depending on the underlying hardware.

3. **Performance Consistency**:
   - The low computation times indicate that the implementation is efficient, leveraging parallelism effectively to handle the matrix-vector multiplication.

# TASK 3: Sparse Matrix Implementation and Testing

### Overview
this task focused on implementing and testing a sparse matrix using the Eigen library. Our goal was to efficiently handle matrix-vector multiplication and save the results to a file.

### Implementation Details
1. **Sparse Matrix Initialization**:
   Using Eigen library to load a sparse matrix from a `.mtx` file, Using Eigen's `loadMarket` function. The matrix was then initialized with a vector of ones to facilitate the matrix-vector multiplication.

2. **Matrix-Vector Multiplication**:
   The core of this task was performing matrix-vector multiplication using the sparse matrix. This operation is crucial for various numerical methods and optimizations. By utilizing Eigen's capabilities, we achieved optimal performance in handling sparse matrices.

3. **Performance Measurement**:
   To evaluate the efficiency of our implementation, we measured the computation time for the matrix-vector multiplication. Using the `chrono` library, recorded the computation time in milliseconds, providing a clear indication of the performance.

4. **Result Storage**:
   The resulting vector from the matrix-vector multiplication was saved to a file in the `.mtx` format using Eigen's `saveMarketVector` function. This step ensures that the results are easily accessible and can be used for further analysis or verification.

### Key Highlights
- **Efficiency**: By using Eigen's sparse matrix capabilities, we achieved efficient matrix-vector multiplication, which is essential for handling large datasets.
- **Performance Tracking**: The inclusion of computation time measurement allows us to monitor and optimize the performance of our implementation.
- **Result Management**: Saving the results to a file is better for large matrices and makes it more suitable to be used in the future for other computations/tests.

### How to Use It
 **in the PMatrix directory**
1. **Compile the code**:
   ```bash
   make
   ```

2. **Run the Sparse Matrix Test**:
   ```bash
   ./main_Pmatrix --sparse sparse_matrix.mtx
   ```

### Example Output

Here’s an example of the output:

```
Implementing with sparse matrix in scalar mode
Loading matrix from file: matrices/spd_test.mtx
Matrix loaded successfully.

Starting matrix-vector multiplication test with matrix size: 48 x 48
Saving result in file: matrices/sparse_result.mtx
Computation time: 0.0021 milliseconds
```

### Matrices used for testing are SPD matrices downloaded from matrix market from Harwell-Boeing collection
1. **Matrix 1**:
   - **Name**: [bcsstk01](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc1/bcsstk01.html) 
   - **Dimensions**: 48 x 48
   - **file name**: sparse_matrix.mtx
  
2. **Matrix 2**:
   - **Name**: [bcsstk14](https://math.nist.gov/MatrixMarket/data/Harwell-Boeing/bcsstruc2/bcsstk14.html)
   - **Dimensions**: 1806 x 1806
   - **file name**: sparse_test.mtx



### Comments on the Results

1. **Final Result Vector**:
   - The result vector is saved in the file `sparse_result.mtx`, confirming that the implementation correctly handles the matrix-vector multiplication and result storage.

2. **Computation Time**:
   - The computation time is recorded and printed, providing a clear indication of the performance of the implementation.

3. **Performance Consistency**:
   - The low computation time indicates that the implementation is efficient, leveraging Eigen's capabilities to handle sparse matrices effectively.

---

## Acknowledgements

This was a hands-on challenge assigned as part of the course *ADVANCED METHODS FOR SCIENTIFIC COMPUTING in High-Performance Computing (2024/25)* at *Politecnico di Milano*. We extend our sincere gratitude to:

- **Professor [Luca Formaggia](https://formaggia.faculty.polimi.it/)** for providing excellent guidance throughout the course.

This project has significantly enhanced our understanding of *numerical methods* and their applications in *high-performance computing*.

---

## Contributors

- [Salvatore Mariano Librici](https://www.linkedin.com/in/salvatore-mariano-librici/)
- [Nadah Khaled](https://www.linkedin.com/in/nadahkhaledd10/)
- [Milica Sanjevic](https://www.linkedin.com/in/milica-sanjevic-321392327/)
- [Hirdesh Kumar](https://www.linkedin.com/in/hirdeshkumar2407/)
- [Martina Ditrani](https://www.linkedin.com/in/martina-ditrani-644507232/)
