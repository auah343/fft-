#include <iostream>
#include "fft.h"

int main() {
    // 创建一个简单的示例矩阵
    Eigen::MatrixXd X(4, 3);
    X << 1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12;


    // 计算 FFT
    Eigen::MatrixXcd Y = FFTLibrary::fft(X);

    std::cout << "FFT of X:\n" << Y << std::endl;

    // 计算 n 点 FFT，指定点数 n = 8
    Eigen::MatrixXcd Y_n = FFTLibrary::fft(X, 8);
    std::cout << "8-point FFT of X:\n" << Y_n << std::endl;

    // 沿第 2 维计算 FFT，指定 n = 8
    Eigen::MatrixXcd Y_dim2 = FFTLibrary::fft(X, 8, 2);
    std::cout << "8-point FFT of X along dimension 2:\n" << Y_dim2 << std::endl;

    return 0;
}