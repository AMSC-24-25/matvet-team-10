#ifndef HH_GC___HH
#define HH_GC___HH

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
#include <numeric> // Include for std::inner_product

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
    // Small SPD matrix
    std::cout << "Testing with a small SPD matrix..." << std::endl;
    size_t small_size = 5;
    DenseMatrix small_A(small_size, small_size);
    small_A.randomFillSPD();
    std::vector<double> small_b(small_size, 1.0);
    std::vector<double> small_x(small_size, 0.0);
    int small_max_iter = 100;
    double small_tol = 1e-6;
    int small_result = CG(small_A, small_x, small_b, small_max_iter, small_tol);
    std::cout << "Result: " << (small_result == 0 ? "Converged" : "Failed")
              << ", Iterations: " << small_max_iter
              << ", Residual: " << small_tol << std::endl;

    // Medium SPD matrix
    std::cout << "\nTesting with a medium SPD matrix..." << std::endl;
    size_t medium_size = 50;
    DenseMatrix medium_A(medium_size, medium_size);
    medium_A.randomFillSPD();
    std::vector<double> medium_b(medium_size, 1.0);
    std::vector<double> medium_x(medium_size, 0.0);
    int medium_max_iter = 500;
    double medium_tol = 1e-6;
    int medium_result = CG(medium_A, medium_x, medium_b, medium_max_iter, medium_tol);
    std::cout << "Result: " << (medium_result == 0 ? "Converged" : "Failed")
              << ", Iterations: " << medium_max_iter
              << ", Residual: " << medium_tol << std::endl;

    // Documenting Results
    std::cout << "\nResults documented for analysis." << std::endl;
}

} // namespace LinearAlgebra

#endif