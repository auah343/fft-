# 设置最低 CMake 版本
cmake_minimum_required(VERSION 3.30)

# 设置项目名称
project(fft-)

# 设置 C++ 标准为 C++11 或更高版本
set(CMAKE_CXX_STANDARD 11)

# 查找 Eigen3 库
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 添加源文件
set(SOURCES
        main.cpp fft.cpp
)

# 添加可执行文件
add_executable(fft- main.cpp fft.cpp)

# 链接 Eigen3 库
target_link_libraries(test1 Eigen3::Eigen)