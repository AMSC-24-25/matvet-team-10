## Task 3: Sparse Matrix Implementation and Testing

### Overview
In this task, we tackled the challenge of implementing and testing a sparse matrix using the Eigen library. Our goal was to efficiently handle matrix-vector multiplication and save the results to a file.

### Implementation Details
1. **Sparse Matrix Initialization**:
   We leveraged the Eigen library to load a sparse matrix from a `.mtx` file, a common format for storing sparse matrices. Using Eigen's `loadMarket` function, we ensured a smooth and efficient initialization process. The matrix was then initialized with a vector of ones to facilitate the matrix-vector multiplication.

2. **Matrix-Vector Multiplication**:
   The core of this task was performing matrix-vector multiplication using the sparse matrix. This operation is crucial for various numerical methods and optimizations. By utilizing Eigen's capabilities, we achieved optimal performance in handling sparse matrices.

3. **Performance Measurement**:
   To evaluate the efficiency of our implementation, we measured the computation time for the matrix-vector multiplication. Using the `<chrono>` library, we recorded the computation time in milliseconds, providing a clear indication of the performance.

4. **Result Storage**:
   The resulting vector from the matrix-vector multiplication was saved to a file in the `.mtx` format using Eigen's `saveMarketVector` function. This step ensures that the results are easily accessible and can be used for further analysis or verification.

### Key Highlights
- **Efficiency**: By using Eigen's sparse matrix capabilities, we achieved efficient matrix-vector multiplication, which is essential for handling large datasets.
- **Performance Tracking**: The inclusion of computation time measurement allows us to monitor and optimize the performance of our implementation.
- **Result Management**: Saving the results to a file as matrix size is very big so saving back in a .mtx format would also make it better for future use.

### Conclusion
This task demonstrated the effective use of the Eigen library for sparse matrix operations, highlighting the importance of efficient data handling and performance measurement in numerical computations. The implementation is robust and provides a solid foundation for further enhancements and applications.
