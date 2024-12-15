# MatVet-CG-feature-2
Matrix vector and conjugate gradient

# CG implementation (function) TASK 1

The CG function plays is the main part of this code.

Aim: To solve the equation Ax=b step by step, where when we are testing A is a symmetric positive definite matrix.

**Inputs:**
A: The matrix of the system.
x: a initial guess for the solution (this will be modified in the course of this process).
b: The right-hand side vector.
max_iter: The maximum number of steps to run before stopping.
tol: This indicates the margin of error that is assumed to be reasonable.

**Outputs:**
It returns ‘0’ if it finds a solution in the stipulated steps and ‘1’ if it fails to do so.
Resets x to the closest approximation of a solution.
Resets tol to the final error.
How It Works:
First calculate the starting error (residual) as r=b−Ax.
Then check whether the value of x that was guessed in the beginning is acceptable by checking if the residual value is found to be within the limits set in the tolerance.
If not, improve x steb by step:
-find the step size alpha-factor, a direction update factor (beta-factor), and a new residual.
-alter the working direction vector p and the solution x.
Finally stop if the given maximum number of steps has been reached or the residual is sufficiently small.

**Parallelization:**
It makes uses of OpenMP to accelerate some of the operations such as finding the residual and updating the vectors by executing them simultaneously on several threads.

**Dense Matrix Class:**
Handles matrix storage in row-major or column-major order.
Used to represent the coefficient matrix A.
The CG function uses DenseMatrix class for matrix and vector multiplication (q=A*p) and accessing elements of A for performing operations such as residual calculations.
This class include randomFillSPD to create symmetric positive definite matrices.
Uses OpenMP to parallelizes matrix and vector multiplication.
Handling symmetric positive definite matrix generation and matrix-vector multiplication.

**Test Suite:**
During testing (runTest), a DenseMatrix is created and filled with random SPD values. This matrix is passed to the CG function, along with a vector b and an initial guess x. The CG function then solves the system using the DenseMatrix to perform mentioned operations.
Demonstrates performance on matrices of various sizes.
Evaluates the effect of storage order and computation parallelism.
