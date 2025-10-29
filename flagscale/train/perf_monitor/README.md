# FlagScale Performance Monitor

## 概述

Performance Monitor 是 FlagScale 的性能监控模块，用于实时跟踪和记录训练过程中的性能指标。

### 主要功能

- **TFLOPS 计算** - 实时计算模型训练的浮点运算速度
- **吞吐量监控** - 跟踪 samples/sec 和 tokens/sec
- **内存追踪** - 监控 GPU 显存使用情况
- **性能分解** - 详细记录前向/反向传播等各阶段耗时
- **文件日志** - 性能数据独立保存到文件，不影响主训练日志
- **多格式输出** - 支持文本日志和 JSON 格式

### 支持的模型

- GPT
- LLaMA (支持 GQA)
- Qwen
- Mixtral (MoE)
- Aquila

## 快速开始

### 1. 使用官方 run.py 启动（推荐）

#### 方法一：通过命令行参数启用

```bash
python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  action=run \
  train.data.data_path=../pile_wikipedia_demo/pile_wikipedia_demo \
  train.system.enable_perf_monitor=true \
  train.system.perf_log_interval=10 \
  train.system.perf_log_dir=./outputs/logs/perf_monitor
```

#### 方法二：使用预配置文件

```bash
# 使用包含性能监控的配置文件
python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  train=7b_with_perf_monitor \
  action=run \
  train.data.data_path=../pile_wikipedia_demo/pile_wikipedia_demo
```

执行后会生成脚本，手动运行：
```bash
bash outputs/logs/scripts/host_0_localhost_run.sh
```

### 2. 在配置文件中启用

在 `examples/aquila/conf/train/7b.yaml` 或其他配置文件的 `system` 部分添加：

```yaml
system:
  # 其他系统配置...

  # 性能监控配置
  enable_perf_monitor: True
  perf_log_interval: 10
  perf_log_dir: ./outputs/logs/perf_monitor
  perf_console_output: False
  perf_memory_tracking: True
  perf_breakdown: False
  perf_max_log_files: 10
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable-perf-monitor` | 启用性能监控 | False |
| `--perf-log-interval N` | 日志记录间隔（步数） | 10 |
| `--perf-log-dir PATH` | 日志文件目录 | logs/perf_monitor |
| `--perf-console-output` | 同时输出到控制台 | False |
| `--perf-memory-tracking` | 启用内存追踪 | True |
| `--perf-breakdown` | 显示详细性能分解 | False |
| `--perf-max-log-files N` | 保留的最大日志文件数 | 10 |

## 日志文件说明

性能监控会生成以下文件：

```
logs/perf_monitor/
├── perf_metrics_20240129_103000.log    # 文本格式的性能日志
├── perf_summary_20240129_103000.json   # JSON 格式的汇总数据
└── perf_realtime.log                    # 实时更新的日志文件
```

### 日志格式示例

**文本日志 (perf_metrics_*.log)**：
```
================================================================================
Performance Monitor Session Started: 2024-01-29 10:30:00
================================================================================
Timestamp            Step     TFLOPS/GPU   TFLOPS     Samples/s    Tokens/s     Time(ms)   Memory(GB)
--------------------------------------------------------------------------------
2024-01-29 10:30:15  10       125.34       1002.72    512.0        1048576      235.5      42.50
2024-01-29 10:30:30  20       128.12       1024.96    520.0        1064960      230.2      42.75
```

**JSON 汇总 (perf_summary_*.json)**：
```json
{
  "session_info": {
    "start_time": "20240129_103000",
    "end_time": "2024-01-29T11:30:00",
    "total_iterations": 100
  },
  "final_statistics": {
    "avg_tflops_per_gpu": 127.5,
    "max_tflops_per_gpu": 135.2,
    "min_tflops_per_gpu": 120.1,
    "avg_throughput_tokens": 1050000,
    "peak_memory_gb": 45.2
  },
  "iteration_logs": [...]
}
```

## 分析工具

### 1. 实时监控

```bash
# 实时查看性能指标
tail -f logs/perf_monitor/perf_realtime.log

# 使用 watch 命令
watch -n 1 tail -20 logs/perf_monitor/perf_realtime.log
```

### 2. Python 分析脚本

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# 读取 JSON 汇总
with open('logs/perf_monitor/perf_summary_*.json') as f:
    data = json.load(f)

# 转换为 DataFrame
df = pd.DataFrame(data['iteration_logs'])

# 绘制 TFLOPS 曲线
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['TFLOPS_per_GPU'])
plt.xlabel('Iteration')
plt.ylabel('TFLOPS per GPU')
plt.title('Training Performance')
plt.show()

# 统计分析
print(f"Average TFLOPS: {df['TFLOPS_per_GPU'].mean():.2f}")
print(f"Peak TFLOPS: {df['TFLOPS_per_GPU'].max():.2f}")
print(f"Throughput: {df['tokens_per_sec'].mean():.0f} tokens/s")
```

## 与 TensorBoard 集成

性能监控可以与 TensorBoard 集成使用：

```bash
# 启动训练并输出到 TensorBoard
python flagscale/train/train_gpt.py \
    --enable-perf-monitor \
    --tensorboard \
    --tensorboard-dir ./tb_logs \
    --train-iters 100

# 查看 TensorBoard
tensorboard --logdir=./tb_logs
```

在 TensorBoard 中可以看到：
- `performance/tflops_per_gpu`
- `performance/tflops_total`
- `performance/samples_per_second`
- `performance/tokens_per_second`
- `memory/current_gb`
- `memory/peak_gb`

## 测试

### 运行测试

```bash
# 1. 结构测试（不需要 torch）
python test_perf_simple.py

# 2. 单元测试（需要 torch）
python test_perf_monitor_unit.py

# 3. 集成测试
./test_perf_monitor.sh
```

## 开发指南

### 模块结构

```
flagscale/train/perf_monitor/
├── __init__.py              # 模块导出
├── perf_metrics.py          # 核心监控类
├── perf_logger.py           # 文件日志系统
├── flops_calculator.py      # FLOPS 计算公式
├── integration.py           # 训练集成助手
├── arguments.py             # 命令行参数
├── hooks.py                 # 训练钩子
└── monitor_example.py       # 使用示例
```

### 添加新模型支持

要添加新模型的 FLOPS 计算支持，修改 `flops_calculator.py`：

```python
def calculate_new_model_flops(self, batch_size, seq_length, ...):
    """计算新模型的 FLOPS"""
    # 添加特定模型的 FLOPS 计算公式
    attention_flops = ...
    ffn_flops = ...
    total_flops = attention_flops + ffn_flops
    return total_flops * 3  # 前向 + 反向 + 梯度
```

### 自定义日志格式

修改 `perf_logger.py` 中的 `log_metrics` 方法来自定义日志格式。

## 常见问题

### Q: 为什么日志文件没有生成？

A: 检查以下几点：
1. 确保添加了 `--enable-perf-monitor` 参数
2. 检查日志目录权限
3. 确认只有 rank 0 进程会写入日志

### Q: 如何降低性能监控的开销？

A: 可以调整以下参数：
- 增大 `--perf-log-interval` 减少日志频率
- 关闭 `--perf-breakdown` 避免详细计时
- 关闭 `--perf-memory-tracking` 如果不需要内存监控

### Q: TFLOPS 计算准确吗？

A: TFLOPS 计算基于理论公式，可能与实际有差异：
- 不包括优化器计算
- 不包括通信开销
- 假设完美的计算利用率

实际 TFLOPS 通常会低于理论值。

## 性能影响

性能监控的开销很小：
- 时间开销：< 0.5% 训练时间
- 内存开销：< 100MB（主要是日志缓存）
- 磁盘空间：约 10-50MB/小时（取决于日志间隔）

## 贡献

欢迎贡献新功能或改进！请提交 PR 到 `douzi` 分支。

## License

与 FlagScale 主项目相同的许可证。