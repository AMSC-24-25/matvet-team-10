## TASK 3: Sparse Matrix Implementation and Testing

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
