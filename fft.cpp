//
// Created by auahs on 2025/1/21.
//
#include "fft.h"

namespace FFTLibrary {

    Eigen::MatrixXcd FFTN::fft(const Eigen::MatrixXcd& X, int n, int dim) const {
        Eigen::FFT<double> fft;
        Eigen::MatrixXcd Y(X.rows(), X.cols());

        if (dim == 1) {
            // 对每列进行 n 点 FFT 计算
            for (int i = 0; i < X.cols(); ++i) {
                // 获取列并转换为复数类型
                Eigen::VectorXd x_col_real = X.col(i).real();

                // 如果输入列的长度小于 n，进行补零
                if (x_col_real.size() < n) {
                    Eigen::VectorXd padded_col(n);
                    padded_col.head(x_col_real.size()) = x_col_real; // 保留原始数据，剩余部分补零
                    x_col_real = padded_col;
                }
                    // 如果输入列的长度大于 n，进行截断
                else if (x_col_real.size() > n) {
                    x_col_real.conservativeResize(n); // 截断为 n 长度
                }
                Eigen::VectorXcd x_col = x_col_real.cast<std::complex<double>>();
                Eigen::VectorXcd y_col(n); // 创建复数输出向量
                fft.fwd(y_col, x_col); // 执行 FFT 计算
                Y.col(i) = y_col; // 将计算结果存入结果矩阵
            }
        } else if (dim == 2) {
            // 对每行进行 n 点 FFT 计算
            for (int i = 0; i < X.rows(); ++i) {
                // 获取行并转换为复数类型
                Eigen::VectorXd x_row_real = X.row(i).real();

                // 如果输入行的长度小于 n，进行补零
                if (x_row_real.size() < n) {
                    Eigen::VectorXd padded_row(n);
                    padded_row.head(x_row_real.size()) = x_row_real; // 保留原始数据，剩余部分补零
                    x_row_real = padded_row;
                }
                    // 如果输入行的长度大于 n，进行截断
                else if (x_row_real.size() > n) {
                    x_row_real.conservativeResize(n); // 截断为 n 长度
                }
                Eigen::VectorXcd x_row = x_row_real.cast<std::complex<double>>();
                Eigen::VectorXcd y_row(n); // 创建复数输出向量
                fft.fwd(y_row, x_row); // 执行 FFT 计算
                Y.row(i) = y_row.transpose(); // 将计算结果存入结果矩阵并转置回行向量
            }
        }

        return Y; // 返回按指定维度计算后的 FFT 结果矩阵
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