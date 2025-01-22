//
// Created by auahs on 2025/1/21.
//
#include "fft.h"

namespace FFTLibrary {

    Eigen::MatrixXcd FFTN::fft(const Eigen::MatrixXcd& X, int n, int dim) const {
        Eigen::FFT<double> fft;
        Eigen::MatrixXcd Y;

        if (dim == 1) {
            // Process columns
            Y.resize(n, X.cols());
            for (int i = 0; i < X.cols(); ++i) {
                Eigen::VectorXcd x_col = X.col(i);
                Eigen::VectorXcd y_col;

                if (x_col.size() != n) {
                    Eigen::VectorXcd padded_col = Eigen::VectorXcd::Zero(n);
                    padded_col.head(std::min<int>(x_col.size(), n)) = x_col.head(std::min<int>(x_col.size(), n));
                    y_col = Eigen::VectorXcd(n);
                    fft.fwd(y_col, padded_col);
                } else {
                    y_col = Eigen::VectorXcd(n);
                    fft.fwd(y_col, x_col);
                }
                Y.col(i) = y_col;
            }
        } else if (dim == 2) {
            // Process rows
            Y.resize(X.rows(), n);
            for (int i = 0; i < X.rows(); ++i) {
                Eigen::VectorXcd x_row = X.row(i);
                Eigen::VectorXcd y_row;

                if (x_row.size() != n) {
                    Eigen::VectorXcd padded_row = Eigen::VectorXcd::Zero(n);
                    padded_row.head(std::min<int>(x_row.size(), n)) = x_row.head(std::min<int>(x_row.size(), n));
                    y_row = Eigen::VectorXcd(n);
                    fft.fwd(y_row, padded_row);
                } else {
                    y_row = Eigen::VectorXcd(n);
                    fft.fwd(y_row, x_row);
                }
                Y.row(i) = y_row.transpose();
            }
        }
        return Y;Y; // 返回按指定维度计算后的 FFT 结果矩阵
    }

    // 公共接口：按列计算
    Eigen::MatrixXcd fft(const Eigen::MatrixXcd& X) {
        FFTN fftN;
        return fftN.fft(X, X.rows(), 1);
    }

    // 公共接口：按列计算，n 点 FFT
    Eigen::MatrixXcd fft(const Eigen::MatrixXcd& X, int n) {
        FFTN fftN;
        return fftN.fft(X, n, 1);  // 默认按列处理
    }

    // 公共接口：按维度和点数计算
    Eigen::MatrixXcd fft(const Eigen::MatrixXcd& X, int n, int dim) {
        FFTN fftN;
        return fftN.fft(X, n, dim);
    }
}