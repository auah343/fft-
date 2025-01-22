//
// Created by auahs on 2025/1/21.
//
#include "fft.h"

namespace FFTLibrary {

    // FFTN：指定维度计算 FFT
    Eigen::MatrixXcd FFTN::fft(const Eigen::MatrixXcd& X, int n, int dim) const {
        Eigen::FFT<double> fft;
        Eigen::MatrixXcd Y(X.rows(), X.cols());

        if (dim == 1) {
            // 对每列进行 n 点 FFT 计算
            for (int i = 0; i < X.cols(); ++i) {
                // 获取列，并将其从实数转换为复数
                Eigen::VectorXd x_col_real = X.col(i).real();
                Eigen::VectorXcd x_col = x_col_real.cast<std::complex<double>>();  // Cast to complex

                // 创建复数输出向量
                Eigen::VectorXcd y_col(n);

                // 执行 FFT
                fft.fwd(y_col, x_col);
                Y.col(i) = y_col;
            }
        } else if (dim == 2) {
            // 对每行进行 n 点 FFT 计算
            for (int i = 0; i < X.rows(); ++i) {
                // 获取行，并将其从实数转换为复数
                Eigen::VectorXd x_row_real = X.row(i).real();
                Eigen::VectorXcd x_row = x_row_real.cast<std::complex<double>>();  // Cast to complex

                // 创建复数输出向量
                Eigen::VectorXcd y_row(n);

                // 执行 FFT
                fft.fwd(y_row, x_row);
                Y.row(i) = y_row;
            }
        }
        return Y;
    }

    // 公共接口：按列计算
    Eigen::MatrixXcd fft(const Eigen::MatrixXcd& X) {
        FFTN fftN;
        return fftN.fft(X, X.cols(), 1);
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