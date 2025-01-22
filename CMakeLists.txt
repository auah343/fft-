cmake_minimum_required(VERSION 3.30)
project(untitled1)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 CONFIG REQUIRED)

add_executable(untitled1 main.cpp fft.cpp)

set(SOURCES
        main.cpp fft.cpp
)

target_link_libraries(untitled1 Eigen3::Eigen)