#ifndef CG_HPP
#define CG_HPP

//*****************************************************************
// Iterative template routine -- CG
//
// CG solves the symmetric positive definite linear
// system Ax=b using the Conjugate Gradient method.
//
// CG follows the algorithm described on p. 15 in the
// SIAM Templates book.
//
// The return value indicates convergence within max_iter (input)
// iterations (0), or no convergence within max_iter iterations (1).
//
// Upon successful return, output arguments have the following values:
//
//        x  --  approximate solution to Ax = b
// max_iter  --  the number of iterations performed before the
//               tolerance was reached
//      tol  --  the residual after the final iteration
//
//*****************************************************************


#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <numeric>
#include <chrono>
#include "denseMatrix.hpp"

namespace LinearAlgebra {

template <class Matrix, class Vector>
int CG(const Matrix &A, Vector &x, const Vector &b, int &max_iter, typename Vector::value_type &tol) {
    using Real = typename Vector::value_type;
    Real resid;
    Vector p(b.size(), 0.0);
    Vector q(b.size(), 0.0);
    Real alpha, beta, rho;
    Real rho_1(0.0);

    Real normb = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
    Vector r = b;
    Vector Ax = A * x;

    #pragma omp parallel for
    for (size_t i = 0; i < b.size(); ++i) {
        r[i] -= Ax[i];
    }

    if (normb == 0.0)
        normb = 1;

    if ((resid = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0)) / normb) <= tol) {
        tol = resid;
        max_iter = 0;
        return 0;
    }

    for (int i = 1; i <= max_iter; i++) {
        rho = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

        if (i == 1)
            p = r;
        else {
            beta = rho / rho_1;
            #pragma omp parallel for
            for (size_t j = 0; j < p.size(); ++j) {
                p[j] = r[j] + beta * p[j];
            }
        }

        q = A * p;
        alpha = rho / std::inner_product(p.begin(), p.end(), q.begin(), 0.0);

        #pragma omp parallel for
        for (size_t j = 0; j < x.size(); ++j) {
            x[j] += alpha * p[j];
            r[j] -= alpha * q[j];
        }

        resid = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0)) / normb;

        if (resid <= tol) {
            tol = resid;
            max_iter = i;
            return 0;
        }

        rho_1 = rho;
    }

    tol = resid;
    return 1;
}

void runTest(size_t size, ORDERING order, int max_iter, double tol) {
    auto start_time = std::chrono::high_resolution_clock::now();
    DenseMatrix A(size, size, order);
    A.randomFillSPD();
    A.printStorageOrder();
    A.printParallelInfo();
    std::vector<double> b(size, 1.0);
    std::vector<double> x(size, 0.0);
    int result = CG(A, x, b, max_iter, tol);
    std::cout << "Result: " << (result == 0 ? "Converged" : "Failed")
              << ", Iterations: " << max_iter
              << ", Residual: " << tol << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto computation_time = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - start_time).count();
    std::cout << "Computation time: " << computation_time << " ms\n";
}

void testCG() {
    std::cout << "\n-------Testing with small SPD matrices: size = 5. ---------------\n";
    runTest(5, ORDERING::ROWMAJOR, 100, 1e-6);
    runTest(5, ORDERING::COLUMNMAJOR, 100, 1e-6);

    std::cout << "\n-------Testing with medium SPD matrices: size = 50. ---------------\n";
    runTest(50, ORDERING::ROWMAJOR, 500, 1e-6);
    runTest(50, ORDERING::COLUMNMAJOR, 500, 1e-6);

    std::cout << "\nResults documented for analysis." << std::endl;
}

} // namespace LinearAlgebra

#endif // CG_HPP