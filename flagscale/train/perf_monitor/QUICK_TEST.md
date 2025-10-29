# 快速测试性能监控功能

## 测试方法 1：使用 + 前缀（立即可用）

```bash
# 在项目根目录执行
python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  action=run \
  train.data.data_path=../data/pile_wikipedia_demo \
  +train.system.enable_perf_monitor=true \
  +train.system.perf_log_interval=10 \
  +train.system.perf_log_dir=./outputs/logs/perf_monitor

# 执行生成的脚本
bash outputs/logs/scripts/host_0_localhost_run.sh
```

## 测试方法 2：使用配置文件（7b.yaml 已更新）

```bash
# 方法 2a: 直接覆盖参数
python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  action=run \
  train.data.data_path=../data/pile_wikipedia_demo \
  train.system.enable_perf_monitor=true

# 方法 2b: 使用预设配置
python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  train=7b_perf \
  action=run \
  train.data.data_path=../data/pile_wikipedia_demo

# 执行生成的脚本
bash outputs/logs/scripts/host_0_localhost_run.sh
```

## 验证是否成功

1. **检查生成的脚本**：
```bash
grep "enable-perf-monitor" outputs/logs/scripts/host_0_localhost_run.sh
# 应该看到 --enable-perf-monitor 参数
```

2. **运行后检查日志**：
```bash
# 查看日志目录
ls -la outputs/logs/perf_monitor/

# 查看实时日志
tail -f outputs/logs/perf_monitor/perf_realtime.log
```

## 期望输出

日志文件格式：
```
================================================================================
Performance Monitor Session Started: 2024-10-30 08:00:00
================================================================================
Timestamp            Step     TFLOPS/GPU   TFLOPS     Samples/s    Tokens/s     Time(ms)   Memory(GB)
--------------------------------------------------------------------------------
2024-10-30 08:00:15  10       125.34       1002.72    512.0        1048576      235.5      42.50
2024-10-30 08:00:30  20       128.12       1024.96    520.0        1064960      230.2      42.75
```

## 故障排查

如果遇到错误：
- `Key 'enable_perf_monitor' is not in struct` → 使用 `+` 前缀
- 没有日志文件生成 → 检查参数是否正确传递，查看生成的脚本
- 权限错误 → 确保输出目录可写