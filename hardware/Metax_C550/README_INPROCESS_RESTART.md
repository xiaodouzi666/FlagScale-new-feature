# In-Process Restart on Metax C550

## Scope

This note documents the current in-process restart path for Metax C550 on `main-legacy`.

Important:

- The active branch is `main-legacy`.
- This feature currently relies on Megatron's built-in in-process restart wrapper.
- No separate Metax-only restart implementation was added in FlagScale.

## Current Code Path

- GPT entrypoint:
  - `flagscale/train/train_gpt.py`
- RWKV entrypoint:
  - `flagscale/train/train_rwkv.py`
- Metax initialize patch:
  - `hardware/Metax_C550/Megatron-LM/megatron/training/initialize.py.patch`

Relevant behavior:

- `train_gpt.py` already calls `inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)`.
- The Metax initialize patch keeps `inprocess_restart.maybe_force_nccl_backend_init(device_id)`.

## Metax-Specific Notes

- The recommended validation path is still the mini Aquila configuration that already passed smoke training.
- Start from a fresh `exp_dir` and force a missing `checkpoint.load`.
- Keep `flash_attn` disabled and use `unfused` attention for the Metax validation path.

## Known Pitfalls

- Do not validate this first on the original full 7B config.
- Do not reuse old checkpoints while testing restart behavior.
- If the YAML does not explicitly define restart keys, pass them with `++`.
- The rank-kill trigger may need to be executed outside the container, depending on how processes are launched.

## Launch Example

```bash
cd /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy

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

## Trigger Method

After iteration logs start appearing, find the worker processes:

```bash
pgrep -af "train_gpt.py"
```

Then kill one local rank, for example:

```bash
pkill -f "train_gpt.py.*local_rank=3"
```

## Expected Signals

Check the main output log for restart-related messages:

```bash
grep -E "InprocessRestarter|RankShouldRestart|NestedRestarter|restart_ex|term_ex" \
  -n /workspace/exp/aquila_inprocess_${TS}/logs/host_0_*.output | tail -n 50
```

Successful recovery usually means:

- restart-related log lines appear
- training does not stall permanently
- iteration logs continue after the restart event

## Recommendation

Use this validation order on Metax:

1. plain mini Aquila cold-start smoke test
2. perf monitor smoke test
3. in-process restart long run with manual rank kill

This keeps the failure surface small and makes restart debugging much easier.
