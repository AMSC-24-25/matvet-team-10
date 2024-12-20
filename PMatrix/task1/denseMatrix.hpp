#ifndef DENSE_MATRIX_HPP
#define DENSE_MATRIX_HPP

#include <iostream>
#include <vector>
#include <omp.h>
#include <random>

enum class ORDERING { ROWMAJOR, COLUMNMAJOR };

template <ORDERING O>
class DenseMatrix {
private:
    std::vector<double> data;
    size_t rows_, cols_;

public:
    DenseMatrix(size_t rows, size_t cols)
        : rows_(rows), cols_(cols), data(rows * cols, 0.0) {}

    double& operator()(size_t i, size_t j) {
        if constexpr (O == ORDERING::ROWMAJOR) {
            return data[j + i * cols_];
        } else {
            return data[i + j * rows_];
        }
    }

    const double& operator()(size_t i, size_t j) const {
        if constexpr (O == ORDERING::ROWMAJOR) {
            return data[j + i * cols_];
        } else {
            return data[i + j * rows_];
        }
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    void randomFillSPD() {
        #pragma omp parallel for
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                double value = static_cast<double>(rand()) / RAND_MAX;
                (*this)(i, j) = value;
                (*this)(j, i) = value;
            }
            (*this)(i, i) += rows_; // Ensure diagonal dominance
        }
    }

    std::vector<double> operator*(const std::vector<double>& x) const {
        std::vector<double> result(rows_, 0.0);
        if constexpr (O == ORDERING::ROWMAJOR) {
            #pragma omp parallel for
            for (size_t i = 0; i < rows_; ++i) {
                for (size_t j = 0; j < cols_; ++j) {
                    result[i] += (*this)(i, j) * x[j];
                }
            }
        } else {
            #pragma omp parallel for
            for (size_t j = 0; j < cols_; ++j) {
                for (size_t i = 0; i < rows_; ++i) {
                    result[i] += (*this)(i, j) * x[j];
                }
            }
        }
        return result;
    }

    void printParallelInfo() const {
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int num_threads = omp_get_num_threads();
            #pragma omp critical
            {
                std::cout << "Thread " << thread_id << " of " << num_threads << " is working.\n";
            }
        }
    }

    void printStorageOrder() const {
        std::cout << "\nMatrix storage order: " << (O == ORDERING::ROWMAJOR ? "Row-major" : "Column-major") << std::endl;
    }
};

#endif // DENSE_MATRIX_HPP
