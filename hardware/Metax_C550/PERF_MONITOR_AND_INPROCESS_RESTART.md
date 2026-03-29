# Metax C550 上的 Perf Monitor 与 In-Process Restart

## 说明

- 当前本地仓库仍然是 legacy runner 架构，实际使用的是：
  - `flagscale/runner/runner_train.py`

新的main分支，由于之前在沐曦机器上测试跑不通，等待后续甲方那边适配好了这边再做同步。

## 当前代码状态

### Perf Monitor

当前 `perf_monitor` 对应的是 legacy runner 里的监控链路：

- `flagscale/runner/elastic/monitor_launcher.py`
- `flagscale/runner/elastic/monitor_service.py`
- `flagscale/runner/elastic/diagnostic.py`

这次已经补了两点：

1. 修复多机场景下 `enable_monitoring` 没有正确透传的问题。
   - 原来 hostfile 路径下 `run_node()` 没把开关传进去，会退回 `_run_each(..., enable_monitoring=True)` 的默认值。
   - 现在已经改成显式透传。

2. 增加了兼容别名，支持直接用 `perf_monitor` 命名启动。
   - `++experiment.runner.enable_perf_monitor=true`
   - `++experiment.runner.perf_monitor_interval=5`

内部实际仍然会映射到 legacy monitor 服务：

- `enable_perf_monitor` -> `enable_monitoring`
- `perf_monitor_interval` -> `monitor_interval`

### 沐曦适配

对 `perf_monitor` 本身不需要额外的 GPU vendor 适配，因为这套逻辑主要做的是：

- 进程状态监控
- 日志收集
- 诊断文件生成

它不依赖 `nvidia-smi`。

这次额外补了 Metax 常见错误关键字，方便诊断报告识别：

- `maca out of memory`
- `mxkw`
- `ioctl create queue block timeout`

### In-Process Restart

当前仓库已经接了 Megatron 自带的 in-process restart 包装，不需要再额外改训练主流程。

已确认的接入点：

- `flagscale/train/train_gpt.py`
- `flagscale/train/train_rwkv.py`
- `hardware/Metax_C550/Megatron-LM/megatron/training/initialize.py.patch`

关键点：

- `train_gpt.py` 已调用 `inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)`
- Metax 的 `initialize.py.patch` 里也保留了：
  - `inprocess_restart.maybe_force_nccl_backend_init(device_id)`

因此，在当前 Metax 训练树里，是否启用 restart，主要取决于传给 Megatron 的运行参数，而不是额外的 runner 改动。

## 进入构建目录

下面所有命令，都应在服务器上的训练构建目录中执行，例如：

```bash
cd /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy
```

## Hydra 参数说明

这里统一建议对新增或不确定是否已在 YAML 中声明的键，使用 `++`。


## 1. Perf Monitor Smoke Test

目标：

- 使用上次已经跑通的 Aquila mini 配置
- 强制冷启动
- 开启 perf monitor
- 快速验证训练链路和 monitor 链路都正常

```bash
TS=$(date +%Y%m%d_%H%M%S)

python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  action=test \
  experiment.exp_dir=/workspace/exp/aquila_perf_smoke_${TS} \
  train.system.checkpoint.load=/workspace/exp/__no_ckpt__/does_not_exist \
  train.system.checkpoint.save=/workspace/exp/aquila_perf_smoke_${TS}/checkpoints \
  train.system.use_flash_attn=false \
  train.model.attention_backend=unfused \
  train.model.num_layers=8 \
  train.model.hidden_size=1024 \
  train.model.num_attention_heads=16 \
  train.model.seq_length=512 \
  train.model.max_position_embeddings=512 \
  train.model.multiple_of=128 \
  train.model.micro_batch_size=1 \
  train.model.global_batch_size=8 \
  train.model.train_samples=16 \
  ++experiment.runner.enable_perf_monitor=true \
  ++experiment.runner.perf_monitor_interval=5
```

### 预期结果

日志中应看到：

- 随机初始化冷启动，不加载旧 checkpoint
- `iteration 1/2`
- `iteration 2/2`
- 成功保存 checkpoint

同时在对应 `exp_dir` 下应出现：

```bash
/workspace/exp/aquila_perf_smoke_${TS}/logs/monitor
```

里面通常会有：

- `status.log`
- `host_*_diagnostic.txt`
- `host_*_current.log`

## 2. Perf Monitor 完整跑法

如果要基于同一套 mini Aquila 配置跑一个更长的任务，可以直接把 `action=test` 改成 `action=run`，并把样本数调大。

下面给一个 200 step 量级的示例：

```bash
TS=$(date +%Y%m%d_%H%M%S)

python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  action=run \
  experiment.exp_dir=/workspace/exp/aquila_perf_run_${TS} \
  train.system.checkpoint.load=/workspace/exp/__no_ckpt__/does_not_exist \
  train.system.checkpoint.save=/workspace/exp/aquila_perf_run_${TS}/checkpoints \
  train.system.use_flash_attn=false \
  train.model.attention_backend=unfused \
  train.model.num_layers=8 \
  train.model.hidden_size=1024 \
  train.model.num_attention_heads=16 \
  train.model.seq_length=512 \
  train.model.max_position_embeddings=512 \
  train.model.multiple_of=128 \
  train.model.micro_batch_size=1 \
  train.model.global_batch_size=8 \
  train.model.train_samples=1600 \
  ++experiment.runner.enable_perf_monitor=true \
  ++experiment.runner.perf_monitor_interval=5
```

说明：

- `train_samples=1600`
- `global_batch_size=8`

因此大约对应：

- `1600 / 8 = 200` iterations

## 3. In-Process Restart 触发测试

### 接入方式

当前建议直接用 Megatron 自带的参数打开：

- `++train.system.inprocess_restart=true`
- `++train.system.inprocess_max_iterations=2`

### 推荐测试方式

对于 Metax，当前最直接的测试方法仍然是：

1. 启动一个比 smoke test 更长的任务
2. 等训练开始打印 iteration
3. 手动杀掉一个 local rank
4. 观察 restarter 日志以及训练是否继续前进

### 启动命令

```bash
TS=$(date +%Y%m%d_%H%M%S)

python run.py \
  --config-path ./examples/aquila/conf \
  --config-name train \
  action=run \
  experiment.exp_dir=/workspace/exp/aquila_inprocess_${TS} \
  train.system.checkpoint.load=/workspace/exp/__no_ckpt__/does_not_exist \
  train.system.checkpoint.save=/workspace/exp/aquila_inprocess_${TS}/checkpoints \
  train.system.use_flash_attn=false \
  train.model.attention_backend=unfused \
  train.model.num_layers=8 \
  train.model.hidden_size=1024 \
  train.model.num_attention_heads=16 \
  train.model.seq_length=512 \
  train.model.max_position_embeddings=512 \
  train.model.multiple_of=128 \
  train.model.micro_batch_size=1 \
  train.model.global_batch_size=8 \
  train.model.train_samples=1600 \
  ++train.system.inprocess_restart=true \
  ++train.system.inprocess_max_iterations=2
```

### 训练启动后，查看 rank 进程

```bash
pgrep -af "train_gpt.py"
```

由于 Aquila 的训练入口仍然是：

```bash
./flagscale/train/train_gpt.py
```

所以这里仍然用 `train_gpt.py` 搜索。

### 手动杀一个 rank

例如杀掉 `local_rank=3`：（容器里面没有这个命令，要在容器外操作）

```bash
pkill -f "train_gpt.py.*local_rank=3"
```

### 观察 restart 日志

```bash
grep -E "InprocessRestarter|RankShouldRestart|NestedRestarter|restart_ex|term_ex" \
  -n /workspace/exp/aquila_inprocess_${TS}/logs/host_0_*.output | tail -n 50
```

如果文件名不是 `host_0_localhost.output`，用通配符即可。

### 预期信号

如果 restart 生效，通常应看到类似信号：

- `RankShouldRestart(...)`
- `InprocessRestarter ... stage=starting`
- `InprocessRestarter ... stage=completed`
- 或 `NestedRestarter ... stage=completed`

更重要的是，后面还要继续出现新的 iteration 日志。

也就是说，判断成功的标准不是只看 restarter 打印，而是：

1. 出现 restart 相关日志
2. 训练没有停死
3. iteration 继续往前走

## 4. 关于 Metax 上的触发方式判断

由于当前远程服务器不可达，这里只能给出“基于现有代码与 NVIDIA 侧测试记录”的判断，不能宣称已经在 Metax 上实测通过。

### 当前判断

Metax 上最值得先试的触发方式仍然是“手动杀单个 local rank”：

- 训练入口与 Megatron in-process wrapper 已经接通
- 当前 C550 路径仍使用 torch distributed + torchrun 启动
- 训练日志中也出现了 NCCL / distributed 初始化相关输出

因此这条路径具备和 NVIDIA 测试类似的基本前提。

## 5. 建议的最小验证顺序

```bash
1. perf monitor smoke
2. perf monitor 长任务
3. in-process restart 手动杀 rank 触发
```