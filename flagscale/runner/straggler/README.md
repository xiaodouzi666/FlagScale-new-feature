# FlagScale Straggler Detection

## 简介
Straggler Detection (掉队者检测) 模块用于在分布式训练过程中监控各个节点和 GPU 的性能表现，检测是否存在明显慢于其他计算单元的“掉队者”（Straggler）。如果有某个 GPU 或节点计算过慢，它将拖慢整个分布式训练的速度。本模块能够实时提供各个节点的算力统计分析日志，帮助用户快速定位性能瓶颈。

## 快速开始

### 单机运行示例 (以 GPT-2 为例)
在运行 `run.py` 时，通过覆盖系统配置参数来开启 Straggler 检测功能：

```bash
python run.py \
  --config-path ./examples/gpt2/conf \
  --config-name train_single \
  action=run \
  +train.system.enable_straggler_detection=true \
  +train.system.straggler_profiling_interval=5 \
  +train.system.straggler_report_interval=10 \
  +train.system.straggler_log_dir=./outputs_gpt2/logs/straggler
```

### 多机运行示例 (以 2机 x 4卡 为例)
多机运行与单机类似，需要在 `runner` 配置中指定 hostfile 和 master 地址等：

```bash
python run.py \
  --config-path ./examples/gpt2/conf \
  --config-name train \
  action=run \
  +train.system.enable_straggler_detection=true \
  +train.system.straggler_profiling_interval=5 \
  +train.system.straggler_report_interval=10 \
  +train.system.straggler_log_dir=./outputs_gpt2/logs/straggler
```
*注意：在报告中，程序会自动标识每个 rank 所在的物理机（通过 `os.environ.get('HOSTNAME')` 或是 `socket.gethostname()` 获取节点名称），以便在多机环境下更好地定位出现问题的机器。*

## 核心参数配置

可通过命令行覆写（如上）或在 yaml 配置文件中修改以下相关参数：

- `enable_straggler_detection` (bool): 是否开启 Straggler 检测（默认: `False`）。
- `straggler_profiling_interval` (int): 采样间隔，即每隔多少 step 记录一次运行耗时数据（默认: `10`）。
- `straggler_report_interval` (int): 报告间隔，即每隔多少 step 自动生成并保存一次统计分析报告（默认: `100`）。
- `straggler_threshold` (float): 判断某节点是否为掉队者的相对延迟阈值（例如 `1.5` 表示比最快节点慢 50% 及以上即被认为是 Straggler，默认: `1.5`）。
- `straggler_log_dir` (str): Straggler 的 JSON 报告文件保存目录。

## 输出报告解读

当到达指定的 `straggler_report_interval` 步数时，代码主要会在 Rank 0 的控制台打印文本报告格式的分析结果，并在指定的 `straggler_log_dir` 目录下定时生成 `.json` 格式的文件。

### 1. 控制台输出示例
```text
=== Straggler Report at Step 10 ===

✓ No stragglers detected.

Section Timings (ms):

  optimizer:
    Min: 9.17ms, Max: 10.12ms, Avg: 9.55ms, Slowdown: 1.10x
    Rank 0: 9.98ms
    Rank 1: 10.12ms
    ...

  forward_backward:
    Min: 281.94ms, Max: 343.19ms, Avg: 314.23ms, Slowdown: 1.22x
    Rank 0: 343.19ms
    Rank 1: 341.51ms
    ...

GPU Performance Scores (higher=faster):
  Rank 0 (p-phy-zy-daxing-kt-lc-a800-node-prod-15-128): 2.9138
  Rank 1 (p-phy-zy-daxing-kt-lc-a800-node-prod-15-128): 2.9282
  ...
```

### 2. JSON 格式报告
保存的类似 `straggler_report_step_10.json` 的文件会汇总详细的数据：
```json
{
  "step": 10,
  "section_scores": {
    "optimizer": {
      "0": 0.004562,
      "1": 0.004560
    },
    "forward_backward": {
      "0": 0.391165,
      "1": 0.391496
    }
  },
  "comm_stats": {},
  "gpu_scores": {
    "0": 2.556464,
    "1": 2.554298
  },
  "straggler_ranks": [],
  "node_names": {
    "0": "p-phy-zy-daxing-kt-lc-a800-node-prod-15-128",
    "1": "p-phy-zy-daxing-kt-lc-a800-node-prod-15-128"
  },
  "timestamp": 1764885816.7089095
}
```

**JSON 字段解读：**
- **`step`**: 本次生成报告的 iteration/step 编号。
- **`section_scores`**: 每个 rank（即每张卡/节点）在指定模块代码段（如 `forward_backward`, `optimizer` 等）的平均耗时，单位为**秒**。
- **`gpu_scores`**: 综合推算出的 GPU 性能总得分。根据各个 section 的反向时长综合得分，数值**越高**说明计算越快。
- **`straggler_ranks`**: 代码根据阈值判断出来的严重掉队卡。如果是 `[]` 则表示各项正常，没有检测到掉队现象；如果有内容（如 `[2, 7]`），则表示对应的 Rank 卡顿严重。
- **`node_names`**: 映射每个 Rank 所属机器名称，对多机多卡训练排查故障卡非常重要，可明确知道当前 Rank 落在哪台服务器上。
- **`timestamp`**: 此报告产出时的时间戳。

## 自定义代码块监控（二次开发使用）

默认情况下，Straggler 模块只监控了训练过程中最核心的 `forward_backward` (前向与反向传播) 和 `optimizer` (参数更新) 这两个部分。

但是，如果你的模型有一些特殊的设计，比如**额外的损失计算**、**自定义的通信操作**、或者是**某些特定的数据预处理逻辑**，你可能会想知道：“这些我写的特殊代码，在不同 GPU 上执行起来有没有掉队的现象？”

这时，你可以使用 Straggler 模块提供的功能，把**你自己写的代码块**也包起来监控。主要有两种比较方便的方式：

### 方式一：使用上下文管理器 `SectionContext`（推荐）
如果你有一段连续的代码需要监控，可以使用 `SectionContext` 将其包裹起来。它会在进入和退出代码块时自动帮你记录时间。

```python
from flagscale.runner.straggler import get_fs_straggler_detector
from flagscale.runner.straggler.section import OptionalSectionContext

# 获取检测器实例
fs_straggler = get_fs_straggler_detector()
enabled = fs_straggler is not None and fs_straggler.is_enabled()

# 使用上下文管理器包裹你需要监控的代码
# "my_custom_module" 是你自己给这段代码起的名字
with OptionalSectionContext(fs_straggler, "my_custom_module", enabled=enabled):
    # 这里放你自己复杂的、想要监控耗时的代码
    # ...
    # outputs = my_special_layer(inputs)
    # ...
```

### 方式二：使用装饰器 `create_section_decorator`
如果你想直接监控某个完整的函数，可以使用装饰器。

```python
from flagscale.runner.straggler import get_fs_straggler_detector
from flagscale.runner.straggler.section import create_section_decorator

fs_straggler = get_fs_straggler_detector()

# 直接在你的函数定义上加这个装饰器
@create_section_decorator(fs_straggler, "my_custom_function")
def my_special_function(data):
    # ... 函数逻辑 ...
    pass
```

通过以上方法添加自定义监控后，只要在启动配置的 `monitor_sections` 列表里加上你定义的名字（例如上面的 `"my_custom_module"` 或 `"my_custom_function"`），这些步骤的耗时也会一样出现在最终生成的 `straggler_report_step_xxx.json` 报告里，和 `optimizer` 它们同列显示！
