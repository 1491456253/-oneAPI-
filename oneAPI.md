# 利用oneAPI Math Kernel Library工具加速计算矩阵和向量乘积
## oneAPI简介
oneAPI是一个开放的、统一的编程模型，它的目标是简化在各种硬件平台（包括CPUs，GPUs，FPGAs等）上进行编程和优化的过程。它由Intel推出，旨在对抗NVIDIA的CUDA编程模型，后者在GPU编程和加速计算方面占有主导地位。oneAPI提供了一系列用于数据并行计算的软件工具，其中包括DPC++编译器、性能库、分析工具和调试工具等。
DPC++（Data Parallel C++）是oneAPI中的核心组件，它是基于现代C++和SYCL标准的一种编程语言，旨在简化在不同类型的硬件上进行数据并行编程。使用DPC++，开发人员可以编写一份代码，然后在各种硬件（CPU、GPU、FPGA等）上运行，而无需针对每种硬件编写不同的代码。
----    
----
## oneAPI Math Kernel Library 简介
oneAPI Math Kernel Library（oneMKL），前身为Intel Math Kernel Library，是oneAPI中的一个组件，提供了一系列数学例程，例如BLAS（基础线性代数子程序）、LAPACK（线性代数程序包）、FFT（快速傅里叶变换）以及随机数生成等功能。oneMKL是针对最高性能设计的，支持多种硬件平台，提供了在各种硬件（包括Intel和非Intel硬件）上运行的优化例程。可以通过oneMKL进行数学计算，从而避免手动实现和优化这些基础运算。

### 本次将使用oneAPI Math Kernel Library来实现加速简单矩阵-向量乘法操作

一下这段代码是一个使用oneAPI Math Kernel Library（oneMKL）实现的简单矩阵-向量乘法操作。

在这段代码首先从用户那里获取一个矩阵和一个向量的输入。接着，我们使用oneMKL中的oneapi::mkl::blas::row_major::gemv函数来进行矩阵-向量乘法。这个函数对应于BLAS中的GEMV（General Matrix Vector multiplication）例程，它对一个矩阵和一个向量进行乘法操作。


```
#include <iostream>
#include <vector>
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
// 获取矩阵输入的函数
std::vector<double> get_matrix(int rows, int cols) {
    std::vector<double> matrix(rows * cols);
    std::cout << "请输入 " << rows * cols << " 个矩阵元素: ";
    for (int i = 0; i < rows * cols; i++) {
        std::cin >> matrix[i];
    }
    return matrix;
}
// 获取向量输入的函数
std::vector<double> get_vector(int size) {
    std::vector<double> vector(size);
    std::cout << "请输入 " << size << " 个向量元素: ";
    for (int i = 0; i < size; i++) {
        std::cin >> vector[i];
    }
    return vector;
}

int main() {
    int rows, cols;
    std::cout << "请输入矩阵的行数和列数: ";
    std::cin >> rows >> cols;

    std::vector<double> A = get_matrix(rows, cols);
    std::vector<double> x = get_vector(cols);
    std::vector<double> y(cols); // y向量，存储结果
    // 选择设备
    sycl::device dev = sycl::device(sycl::default_selector());
    // 创建队列
    sycl::queue queue(dev);
    try {
        // 计算矩阵-向量乘积
        oneapi::mkl::blas::row_major::gemv(
            queue,
            oneapi::mkl::transpose::nontrans,
            rows, cols,
            1.0,
            A.data(), cols,
            x.data(), 1,
            0.0,
            y.data(), 1
        );
        // 等待命令队列完成操作
        queue.wait_and_throw();
        // 打印结果
        for (int i = 0; i < cols; i++) {
            std::cout << y[i] << ' ';
        }
        std::cout << '\n';
    } catch(sycl::exception const &e) {
        // 捕获主机代码中的异常
        std::cout << "捕获到同步SYCL异常:\n"
                  << e.what() << '\n';
        return 1;
    }
    return 0;
}

```

使用oneMKL可以带来许多优势。首先，oneMKL利用了现代处理器的并行处理能力，所以其提供的函数通常比直接实现更快。此外，oneMKL还针对Intel处理器进行了优化，因此在Intel硬件上运行时可以获得最佳性能。另外，由于oneMKL是oneAPI的一部分，因此它支持异构计算，即可以在CPU和GPU等不同的设备上运行。

在这段代码中，oneMKL帮助我们以一种简洁、高效的方式实现了矩阵-向量乘法。我们只需要一行代码，就可以调用oneapi::mkl::blas::row_major::gemv函数。并且，我们不需要关心如何优化这个计算，因为这已经由oneMKL处理。

总的来说，使用oneMKL可以使开发人员以一种简单、高效的方式进行复杂的数学计算，而不需要关心底层的优化和并行化。这使我们可以将注意力集中在算法的逻辑上，而不是优化计算。