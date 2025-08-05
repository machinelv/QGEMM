# CUDA GEMM Benchmark Test Framework

这是一个纯CUDA的GEMM性能测试框架，不依赖任何Python组件。该框架基于CUTLASS库实现，专门针对SM80+架构的GPU进行优化。

## 特性

- **纯CUDA实现**: 无Python依赖，完全使用C++/CUDA实现
- **多精度支持**: 支持BF16和INT8精度
- **性能基准测试**: 提供TFLOPS和带宽测量
- **正确性验证**: 自动验证GEMM结果的正确性
- **易于扩展**: 支持添加自定义GEMM实现
- **现代C++设计**: 使用模板和现代C++特性

## 项目结构

```
tests/
├── CMakeLists.txt          # 专用CMake构建配置
├── build.sh               # 自动化构建脚本
├── cuda_benchmark.cpp     # 主测试程序
├── includes/
│   ├── test_config.h      # 测试配置和基准测试类
│   └── simple_timer.h     # CUDA事件计时器
└── README.md             # 本文档
```

## 系统要求

### 硬件要求
- NVIDIA GPU with Compute Capability 8.0+ (Ampere架构或更新)
- 支持的GPU: RTX 3080/3090, RTX 4080/4090, A100, H100等

### 软件要求
- CUDA Toolkit 11.8 或更新版本
- CMake 3.21 或更新版本
- C++17 兼容的编译器
- CUTLASS库 (推荐使用最新版本)

## 安装与构建

### 1. 安装CUTLASS

如果您还没有安装CUTLASS，可以从以下位置获取：

```bash
# 克隆CUTLASS仓库
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="80;89"
make -j8
sudo make install
```

或者，如果您的项目中已经包含了CUTLASS（如在ref/cutlass/目录下），CMake会自动找到它。

### 2. 构建测试框架

```bash
# 进入tests目录
cd tests

# 给构建脚本执行权限
chmod +x build.sh

# 运行构建脚本
./build.sh
```

### 手动构建（可选）

如果您偏好手动构建：

```bash
# 创建并进入构建目录
mkdir build && cd build

# 配置CMake
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="80;89"

# 构建
make -j$(nproc)
```

## 运行测试

构建完成后，可执行文件位于 `build/bin/cuda_benchmark`：

```bash
cd build
./bin/cuda_benchmark
```

## 测试输出

程序会输出详细的性能测试结果：

```
CUDA SM80 GEMM Benchmark using CUTLASS
Testing precisions: BF16, INT8
GPU: NVIDIA GeForce RTX 4090
Compute Capability: 8.9

=== GEMM Benchmark ===
Problem size: M=512, N=512, K=512
Precision: bf16
Warmup runs: 5
Benchmark runs: 20
==========================================================================================
Function                 Time(ms)    TFLOPS      Bandwidth(GB/s) Correctness Max Error
------------------------------------------------------------------------------------------
cutlass_gemm_test        0.123       2.184       145.2           PASS        1.23e-03
```

## 代码架构

### 核心类

1. **TestConfig**: 测试配置结构
   - 定义矩阵尺寸、精度、迭代次数等参数

2. **GEMMBenchmark**: 主要的基准测试类
   - 模板化设计，支持不同的输入/输出类型
   - 提供内存管理、数据初始化、性能测量等功能

3. **Timer**: 高精度计时器
   - 支持CUDA事件和CPU计时两种模式
   - 自动选择最适合的计时方式

### 扩展自定义GEMM实现

您可以轻松添加自己的GEMM实现：

```cpp
// 定义您的GEMM函数
template<typename T>
void my_custom_gemm(T* A, size_t ldA, T* B, size_t ldB, 
                    T* C, size_t ldC, size_t M, size_t N, size_t K) {
    // 您的GEMM实现
}

// 在benchmark中添加
benchmark.add_custom_function("my_custom_gemm", my_custom_gemm<float>);
```

## 配置选项

### CMAKE选项

- `CMAKE_CUDA_ARCHITECTURES`: 目标GPU架构 (默认: "80;89")
- `CMAKE_BUILD_TYPE`: 构建类型 (建议: Release)
- `CUTLASS_INCLUDE_DIR`: CUTLASS头文件路径 (自动检测)

### 测试配置

在 `cuda_benchmark.cpp` 中修改以下配置：

```cpp
// 修改测试矩阵尺寸
std::vector<TestConfig> configs_bf16 = {
    TestConfig(1024, 1024, 1024, "bf16"),
    TestConfig(2048, 2048, 2048, "bf16"),
    // 添加更多尺寸...
};

// 修改预热和基准测试次数
TestConfig(M, N, K, precision, warmup_runs, benchmark_runs, tolerance, seed)
```

## 故障排除

### 常见问题

1. **"CUTLASS not found" 错误**
   - 确保CUTLASS已正确安装
   - 设置 `CUTLASS_INCLUDE_DIR` 环境变量

2. **编译错误相关的架构**
   - 确保您的GPU支持指定的CUDA架构
   - 修改 `CMAKE_CUDA_ARCHITECTURES` 以匹配您的GPU

3. **运行时错误**
   - 检查CUDA驱动程序版本
   - 确保GPU内存足够大

### 调试模式

使用Debug构建获取更多信息：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

## 性能调优建议

1. **矩阵尺寸**: 使用256的倍数通常能获得更好的性能
2. **内存对齐**: 确保数据对齐到适当的边界
3. **GPU预热**: 增加预热轮数以获得更稳定的测量结果
4. **系统负载**: 在测试期间最小化其他GPU使用

## 贡献

欢迎提交改进建议和bug报告。在提交之前，请确保：

1. 代码遵循现有的编码风格
2. 添加适当的注释和文档
3. 测试新功能的正确性和性能

## 许可证

本项目遵循与主项目相同的许可证。
