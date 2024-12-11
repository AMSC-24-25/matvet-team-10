#ifndef HH_GC___HH
#define HH_GC___HH

//***********************
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
//***********************

#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <numeric> // Include this header for std::inner_product

namespace LinearAlgebra {

class DenseMatrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows_, cols_;

public:
    DenseMatrix(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    double& operator()(size_t i, size_t j) {
        return data[i][j];
    }

    const double& operator()(size_t i, size_t j) const {
        return data[i][j];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    void randomFillSPD() {
        #pragma omp parallel for
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double value = static_cast<double>(rand()) / RAND_MAX;
                data[i][j] = value;
                data[j][i] = value;
            }
            data[i][i] += rows_; // Ensure diagonal dominance
        }
    }

    std::vector<double> operator*(const std::vector<double>& x) const {
        std::vector<double> result(rows_, 0.0);
        #pragma omp parallel for
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result[i] += data[i][j] * x[j];
            }
        }
        return result;
    }
};

template <class Vector>
int CG(const DenseMatrix &A, Vector &x, const Vector &b, int &max_iter, typename Vector::value_type &tol) {
    using Real = typename Vector::value_type;
    Real   resid;
    Vector p(b.size(), 0.0);
    Vector q(b.size(), 0.0);
    Real   alpha, beta, rho;
    Real   rho_1(0.0);

    Real   normb = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
    Vector r = b;
    Vector Ax = A * x;

    #pragma omp parallel for
    for (size_t i = 0; i < b.size(); ++i) {
        r[i] -= Ax[i];
    }

    if(normb == 0.0)
        normb = 1;

    if((resid = std::sqrt(std::inner_product(r.begin(), r.end(), r.begin(), 0.0)) / normb) <= tol) {
        tol = resid;
        max_iter = 0;
        return 0;
    }

    for(int i = 1; i <= max_iter; i++) {
        rho = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

        if(i == 1)
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

        if(resid <= tol) {
            tol = resid;
            max_iter = i;
            return 0;
        }

        rho_1 = rho;
    }

    tol = resid;
    return 1;
}

void testCG() {
    size_t n = 10; // Size of the matrix (medium size for testing)
    DenseMatrix A(n, n);
    A.randomFillSPD();

    std::vector<double> b(n, 1.0); // Right-hand side
    std::vector<double> x(n, 0.0); // Initial guess

    int max_iter = 1000;
    double tol = 1e-6;

    int result = CG(A, x, b, max_iter, tol);

    if (result == 0) {
        std::cout << "Converged in " << max_iter << " iterations with residual " << tol << "\n";
    } else {
        std::cout << "Failed to converge in " << max_iter << " iterations. Final residual: " << tol << "\n";
    }
}

} // namespace LinearAlgebra

#endif