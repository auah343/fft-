#ifndef FFT_LIBRARY_H
#define FFT_LIBRARY_H

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include <vector>

namespace FFTLibrary {

    // FFT 基类
    class FFTBase {
    public:
        virtual ~FFTBase() = default;

        // 纯虚函数，按需重写 FFT 计算
        virtual Eigen::MatrixXcd fft(const Eigen::MatrixXcd& X, int n = -1, int dim = -1) const = 0;
    };

    // FFT 指定维度计算
    class FFTN : public FFTBase {
    public:
        Eigen::MatrixXcd fft(const Eigen::MatrixXcd& X, int n = -1, int dim = -1) const override;
    };

    // 计算 FFT 的公共接口
    Eigen::MatrixXcd fft(const Eigen::MatrixXcd& X);
    Eigen::MatrixXcd fft(const Eigen::MatrixXcd& X, int n);
    Eigen::MatrixXcd fft(const Eigen::MatrixXcd& X, int n, int dim);

}

#endif // FFT_LIBRARY_H
