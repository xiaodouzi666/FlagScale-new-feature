# GPT-2 Runner 功能测试

这份文档整理了当前 latest 分支下，基于 GPT-2 示例验证 runner 相关功能的命令。

## 前置条件

- 仓库根目录：`/workspace/FlagScale-new-feature`
- Conda 环境：`flagscale-train`
- 数据文件：
  - `/workspace/FlagScale-new-feature/data/pile_wikipedia_demo.bin`
  - `/workspace/FlagScale-new-feature/data/pile_wikipedia_demo.idx`
- tokenizer 文件：
  - `/workspace/FlagScale-new-feature/examples/gpt2/tokenizer/vocab.json`
  - `/workspace/FlagScale-new-feature/examples/gpt2/tokenizer/merges.txt`

## Straggler

### Smoke Test：正常测试

```bash
cd /workspace/FlagScale-new-feature
source /root/miniconda3/bin/activate flagscale-train

rm -rf /workspace/FlagScale-new-feature/outputs_straggler_smoke_normal

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  /workspace/FlagScale-new-feature/tools/straggler/straggler_smoke.py \
  --steps 12 \
  --profiling-interval 5 \
  --report-interval 10 \
  --threshold 1.5 \
  --output-dir /workspace/FlagScale-new-feature/outputs_straggler_smoke_normal
```

### Smoke Test：手动 trigger

先启动外部 burner：

```bash
cd /workspace/FlagScale-new-feature
source /root/miniconda3/bin/activate flagscale-train

nohup bash -lc 'CUDA_VISIBLE_DEVICES=7 python /workspace/FlagScale-new-feature/tools/straggler/gpu_burner.py --size 12288' \
  >/tmp/fs_straggler_gpu7_burn.log 2>&1 & echo $! >/tmp/fs_straggler_gpu7_burn.pid
```

再运行 smoke test：

```bash
cd /workspace/FlagScale-new-feature
source /root/miniconda3/bin/activate flagscale-train

rm -rf /workspace/FlagScale-new-feature/outputs_straggler_smoke_trigger

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  /workspace/FlagScale-new-feature/tools/straggler/straggler_smoke.py \
  --steps 8 \
  --profiling-interval 1 \
  --report-interval 5 \
  --threshold 1.2 \
  --output-dir /workspace/FlagScale-new-feature/outputs_straggler_smoke_trigger
```

测试结束后关闭 burner：

```bash
kill "$(cat /tmp/fs_straggler_gpu7_burn.pid)"
rm -f /tmp/fs_straggler_gpu7_burn.pid
```

### Runner 正式测试：正常测试

```bash
cd /workspace/FlagScale-new-feature
source /root/miniconda3/bin/activate flagscale-train

rm -rf /workspace/FlagScale-new-feature/outputs_gpt2/checkpoints/*
rm -rf /workspace/FlagScale-new-feature/outputs_gpt2/logs/straggler/*

python run.py \
  --config-path ./examples/gpt2/conf \
  --config-name train_single \
  action=run \
  ++experiment.task.entrypoint=flagscale/train/megatron/train_gpt.py \
  ++experiment.runner.master_port=25124 \
  train.data.data_path=/workspace/FlagScale-new-feature/data/pile_wikipedia_demo \
  +train.system.enable_straggler_detection=true \
  +train.system.straggler_profiling_interval=5 \
  +train.system.straggler_report_interval=10 \
  +train.system.straggler_warmup_steps=0 \
  +train.system.straggler_log_dir=./outputs_gpt2/logs/straggler
```

### Runner 正式测试：手动 trigger

先启动上面同一个 burner，然后运行：

```bash
cd /workspace/FlagScale-new-feature
source /root/miniconda3/bin/activate flagscale-train

rm -rf /workspace/FlagScale-new-feature/outputs_gpt2/checkpoints/*
rm -rf /workspace/FlagScale-new-feature/outputs_gpt2/logs/straggler/*

python run.py \
  --config-path ./examples/gpt2/conf \
  --config-name train_single \
  action=run \
  ++experiment.task.entrypoint=flagscale/train/megatron/train_gpt.py \
  ++experiment.runner.master_port=25124 \
  train.data.data_path=/workspace/FlagScale-new-feature/data/pile_wikipedia_demo \
  +train.system.enable_straggler_detection=true \
  +train.system.straggler_profiling_interval=1 \
  +train.system.straggler_report_interval=5 \
  +train.system.straggler_threshold=1.2 \
  +train.system.straggler_warmup_steps=0 \
  +train.system.straggler_log_dir=./outputs_gpt2/logs/straggler
```

## Perf Monitor

### Smoke Test

```bash
cd /workspace/FlagScale-new-feature
source /root/miniconda3/bin/activate flagscale-train

rm -rf /workspace/FlagScale-new-feature/outputs_perf_smoke

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  /workspace/FlagScale-new-feature/tools/perf_monitor/perf_smoke.py \
  --steps 12 \
  --log-interval 5 \
  --output-dir /workspace/FlagScale-new-feature/outputs_perf_smoke
```

### Runner 正式测试

```bash
cd /workspace/FlagScale-new-feature
source /root/miniconda3/bin/activate flagscale-train

rm -rf /workspace/FlagScale-new-feature/outputs_gpt2/logs/perf_monitor/*

python run.py \
  --config-path ./examples/gpt2/conf \
  --config-name train_single \
  action=run \
  ++experiment.task.entrypoint=flagscale/train/megatron/train_gpt.py \
  ++experiment.runner.master_port=25125 \
  train.data.data_path=/workspace/FlagScale-new-feature/data/pile_wikipedia_demo \
  +train.system.enable_perf_monitor=true \
  +train.system.perf_log_interval=5 \
  +train.system.perf_console_output=true \
  +train.system.perf_log_dir=./outputs_gpt2/logs/perf_monitor
```

## In-Process Restart

### Smoke Test

```bash
cd /workspace/FlagScale-new-feature
source /root/miniconda3/bin/activate flagscale-train

rm -rf /workspace/FlagScale-new-feature/outputs_gpt2_inprocess_smoke
mkdir -p /workspace/FlagScale-new-feature/outputs_gpt2_inprocess_smoke
unset FT_SIM_FAULT_DESC

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  /workspace/FlagScale-new-feature/flagscale/train/megatron/train_gpt.py \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 4 \
  --use-distributed-optimizer \
  --fp16 --initial-loss-scale 65536 --min-loss-scale 1.0 \
  --attention-softmax-in-fp32 --accumulate-allreduce-grads-in-fp32 \
  --num-layers 8 --hidden-size 1024 --num-attention-heads 16 \
  --seq-length 512 --max-position-embeddings 512 --init-method-std 0.02 \
  --attention-dropout 0.0 --hidden-dropout 0.0 --weight-decay 0.1 --clip-grad 1.0 \
  --train-iters 200 --eval-iters 0 --eval-interval 10000 \
  --micro-batch-size 1 --global-batch-size 16 --seed 1234 \
  --adam-beta1 0.9 --adam-beta2 0.95 --lr 1e-4 --min-lr 1e-5 \
  --lr-warmup-iters 1 --lr-decay-style cosine \
  --data-path /workspace/FlagScale-new-feature/data/pile_wikipedia_demo \
  --split 969,30,1 \
  --legacy-tokenizer --tokenizer-type GPT2BPETokenizer \
  --vocab-file /workspace/FlagScale-new-feature/examples/gpt2/tokenizer/vocab.json \
  --merge-file /workspace/FlagScale-new-feature/examples/gpt2/tokenizer/merges.txt \
  --vocab-size 50257 \
  --save /workspace/FlagScale-new-feature/outputs_gpt2_inprocess_smoke/checkpoints \
  --load /workspace/FlagScale-new-feature/outputs_gpt2_inprocess_smoke/checkpoints \
  --tensorboard-dir /workspace/FlagScale-new-feature/outputs_gpt2_inprocess_smoke/tensorboard \
  --log-interval 1 --save-interval 1000 \
  --inprocess-restart --inprocess-max-iterations 2 \
  2>&1 | tee /workspace/FlagScale-new-feature/outputs_gpt2_inprocess_smoke/inprocess.output
```

当日志里开始出现 `iteration` 后，手动杀一个 rank：

```bash
pgrep -af "train_gpt.py"
pkill -f "train_gpt.py.*local_rank=3"
grep -E "InprocessRestarter|RankShouldRestart" -n /workspace/FlagScale-new-feature/outputs_gpt2_inprocess_smoke/inprocess.output | tail -n 50
```

### Runner 正式测试

```bash
cd /workspace/FlagScale-new-feature
source /root/miniconda3/bin/activate flagscale-train

unset FT_SIM_FAULT_DESC

python run.py \
  --config-path ./examples/gpt2/conf \
  --config-name train_single \
  action=run \
  ++experiment.task.entrypoint=flagscale/train/megatron/train_gpt.py \
  ++train.system.inprocess_restart=true \
  ++train.system.inprocess_max_iterations=2 \
  ++train.system.use_flash_attn=false \
  ++train.model.train_iters=200 \
  ++experiment.runner.master_port=29523 \
  ++experiment.runner.enable_gpu_health_check=false \
  ++experiment.runner.enable_monitoring=false \
  train.data.data_path=/workspace/FlagScale-new-feature/data/pile_wikipedia_demo
```

当日志里开始出现 `iteration` 后，手动杀一个 rank：

```bash
pgrep -af "train_gpt.py"
pkill -f "train_gpt.py.*local_rank=3"
grep -E "InprocessRestarter|RankShouldRestart" -n /workspace/FlagScale-new-feature/outputs_gpt2/logs/host_0_localhost.output | tail -n 50
```
