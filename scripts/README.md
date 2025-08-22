# Benchmark to Markdown Converter

这个脚本将 CUDA GEMM 基准测试的输出转换为两个 Markdown 格式的表格：
- **TFLOPS Performance Table**: 显示各函数在不同问题规模下的 TFLOPS 性能
- **Speedup Performance Table**: 显示各函数相对于参考函数的加速比

## 使用方法

### 方法 1: 管道输入（推荐）
```bash
./build/bin/cuda_benchmark | python scripts/benchmark_to_md.py
```

### 方法 2: 从文件输入
```bash
python scripts/benchmark_to_md.py < output.txt
```

### 方法 3: 使用 cat 管道
```bash
cat output.txt | python scripts/benchmark_to_md.py
```

### 方法 4: 保存到文件
```bash
./build/bin/cuda_benchmark | python scripts/benchmark_to_md.py > benchmark_results.md
```

## 输出格式

脚本会生成包含以下内容的 Markdown 文档：

1. **TFLOPS Performance Table**: 每个函数在各问题规模下的 TFLOPS 值
2. **Speedup Performance Table**: 每个函数相对于参考函数的加速比（如 1.25x）
3. **Summary Section**: 测试统计信息和最佳性能函数

## 表格格式

表格的列标题为所有测试的函数名，行标题为问题规模（如 "M=512, N=512, K=512"）。

### 示例输出

```markdown
## TFLOPS Performance

| Problem Size | GEMM_kernel_v1 | cublas_gemm_test | cutlass_gemm_test |
| --- | --- | --- | --- |
| M=512, N=512, K=512 | 4.391 | 1.817 | 2.270 |
| M=1024, N=512, K=2048 | 9.292 | 8.189 | 4.753 |

## Speedup Performance

| Problem Size | GEMM_kernel_v1 | cublas_gemm_test | cutlass_gemm_test |
| --- | --- | --- | --- |
| M=512, N=512, K=512 | 2.45x | 1.02x | 1.27x |
| M=1024, N=512, K=2048 | 1.13x | 1.00x | 0.58x |
```

## 特性

- ✅ 支持管道输入
- ✅ 自动解析所有函数和问题规模
- ✅ 生成格式化的 Markdown 表格
- ✅ 包含性能摘要和统计信息
- ✅ 错误处理和用户友好的错误消息
- ✅ 识别最佳性能函数

## 依赖

- Python 3.6+
- 标准库（无需额外安装包）

## 错误处理

如果没有提供输入数据，脚本会显示使用说明并退出。
如果输入数据格式不正确，脚本会报告错误并提供诊断信息。
