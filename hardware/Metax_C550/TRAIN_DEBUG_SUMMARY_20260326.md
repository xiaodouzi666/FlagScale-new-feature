# Metax C550 训练排查总结（2026-03-26）

## 1. 文档范围

本文档用于完整记录围绕 `FlagScale` 在 `Metax_C550` 上进行训练验证的全过程。

本次排查的目标是确认：

- `main-legacy` 分支是否能够在 MetaX C550 服务器上真正跑通训练链路
- 当前失败分别属于环境问题、配置问题、代码兼容问题，还是模型本身问题
- 最后是否已经形成一个可复现的最小 smoke test

本文档包含以下内容：

- 本地仓库和远端环境基线
- 过程中识别出的关键仓库信息
- 从头到尾实际走过的排查路径
- 已执行或最终修正后的命令
- 每一阶段出现的主要报错、根因和处理方式
- 最终成功跑通的配置和结果

说明：

- 下面展示的命令统一使用“最终修正后的正确版本”。
- 如果某一步是人工复制文件或临时手工改文件，而不是完整保留的 shell 命令，文档中会明确说明。
- 除特别注明外，本文中的运行命令、路径、报错都指远端训练服务器。

## 2. 仓库与环境基线

### 2.1 本地仓库

本地工作目录：

```bash
/Users/liujunjun/Desktop/All files in Desktop/muxi/FlagScale
```

当前分支：

```bash
main-legacy
```

本次排查中确认过、且后续判断中起关键作用的仓库文件有：

- `hardware/Metax_C550/Megatron-LM/diff.yaml`
  - 这里记录了 Metax 补丁所基于的 Megatron-LM 基线 commit
- `hardware/Metax_C550/FlagScale/examples/qwen3/conf/train.yaml.patch`
- `hardware/Metax_C550/FlagScale/examples/qwen3/conf/train/10b.yaml.patch`
- `README.md`
  - 后续确认 `pile_wikipedia_demo` 这份 demo 数据实际上是与 Aquila 示例配套的，这一点非常关键

### 2.2 远端构建目录

训练 build 成功后，后续所有运行时排查都发生在下面这个目录：

```bash
/workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy
```

conda 环境：

```bash
flagscale-train
```

### 2.3 远端设备与运行时事实

远端基础环境如下：

- 8 张 MetaX C550
- MACA 安装根目录：

```bash
/opt/maca-3.2.1
```

- 训练时使用的 Python：

```bash
/opt/conda/bin/python3.10
```

### 2.4 调试期间手工验证过的运行时环境

在 launcher 配置尚未完全修正之前，曾经先在 shell 里手工导出环境变量，验证底层运行时是否至少可以正确导入。

当时验证通过的环境变量块如下：

```bash
export MACA_HOME=/opt/maca-3.2.1
export MACA_PATH=/opt/maca-3.2.1
export CUCC_PATH=/opt/maca-3.2.1/tools/cu-bridge
export CUDA_PATH=/opt/maca-3.2.1/tools/cu-bridge
export DEVINFO_ROOT=/opt/maca-3.2.1
export MACA_CLANG=/opt/maca-3.2.1/mxgpu_llvm
export MACA_CLANG_PATH=/opt/maca-3.2.1/mxgpu_llvm/bin
export LD_LIBRARY_PATH=/opt/maca-3.2.1/lib:/opt/maca-3.2.1/mxgpu_llvm/lib:/opt/mxdriver/lib:/opt/maca-3.2.1/ompi/lib:/opt/maca-3.2.1/ucx/lib:${LD_LIBRARY_PATH}
export PATH=/opt/conda/bin:/opt/conda/condabin:/opt/maca-3.2.1/tools/cu-bridge:/opt/maca-3.2.1/bin:/opt/maca-3.2.1/mxgpu_llvm/bin:/opt/maca-3.2.1/ompi/bin:/opt/maca-3.2.1/ucx/bin:/opt/mxdriver/bin:${PATH}
```

用来验证导入的命令是：

```bash
python -c "import transformer_engine; print(transformer_engine.__file__)"
```

这一步很重要，因为它说明：

- 手工 shell 环境下，运行时本身并不是完全坏的
- 后续很多失败，本质上是 launcher 子进程没有继承到这些环境变量，而不是 `transformer_engine` 或 MACA 根本无法使用

## 3. 排查过程总览

### 3.1 第一阶段：本地 `unpatch` 因网络失败

最开始的目标，是先把 Metax C550 对应的训练树展开出来。使用的命令是：

```bash
python tools/patch/unpatch.py --backend Megatron-LM FlagScale --task train --device-type Metax_C550
```

在本地 Mac 上，这一步没有成功。原因不是代码本身立即报补丁错误，而是执行过程中需要访问 GitHub，当前机器网络不通，导致拉取相关内容失败。

当时的结论是：

- 这个失败不足以说明 Metax 的训练 patch 本身有问题
- 更合理的下一步，是在一台网络正常的远端服务器上执行同样的命令

### 3.2 第二阶段：远端 `unpatch` 成功

在远端服务器上，执行了同样的命令：

```bash
python tools/patch/unpatch.py --backend Megatron-LM FlagScale --task train --device-type Metax_C550
```

这次成功了，构建结果位于：

```bash
/workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy
```

这一步非常关键，因为它证明了：

- `main-legacy` 这套 Metax C550 训练树是可以正常生成的
- 至少 build 阶段并没有根本性断裂

### 3.3 第三阶段：先试 Qwen3，但很快被 tokenizer 路径卡住

最早的一条运行路径先试了 Qwen3 的配置。

这条线很快就放弃了，因为它在 tokenizer 路径阶段就失败了，报错核心是本地路径被当成了 Hugging Face repo id：

```text
HFValidationError: Repo id must be in the form ...
'/models/qwentokenizer/'
```

这里的结论是：

- 这条线还没有跑到真正训练逻辑
- 当前错误更像是配置/路径问题，不适合拿来验证底层训练链路


### 3.4 第四阶段：切到 GPT2，暴露出一系列环境与代码问题

Qwen3 提前卡死之后，转而试了 GPT2 路径。

GPT2 最终没有成为成功的 smoke test 路线，但它非常有价值，因为它把很多“与模型无关”的基础问题都提前暴露出来了。

#### 3.4.1 手工复制到远端 build 目录的文件

在 GPT2 路线中，曾手工把以下文件复制到远端 build 树：

- `examples/gpt2/conf/train_single.yaml`
- `examples/gpt2/conf/train.yaml`
- `examples/gpt2/conf/train/small.yaml`
- `examples/gpt2/tokenizer/vocab.json`
- `examples/gpt2/tokenizer/merges.txt`
- `data/pile_wikipedia_demo.bin`
- `data/pile_wikipedia_demo.idx`

这些属于人工文件复制操作，不是完整保留的统一命令块，因此这里只记录结果，不强行伪造命令。

#### 3.4.2 GPT2 配置修正命令

GPT2 路径中，最终确认过、且逻辑上正确的一组修正命令如下：

```bash
cd /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy

sed -i 's|source /opt/conda/bin/activate base|source /opt/conda/bin/activate flagscale-train|' \
  examples/gpt2/conf/train_single.yaml

grep -q 'MACA_HOME:' examples/gpt2/conf/train_single.yaml || \
sed -i '/CUDA_DEVICE_MAX_CONNECTIONS: 1/a\
    MACA_HOME: /opt/maca-3.2.1\
    MACA_PATH: /opt/maca-3.2.1\
    CUCC_PATH: /opt/maca-3.2.1/tools/cu-bridge\
    CUDA_PATH: /opt/maca-3.2.1/tools/cu-bridge\
    DEVINFO_ROOT: /opt/maca-3.2.1\
    MACA_CLANG: /opt/maca-3.2.1/mxgpu_llvm\
    MACA_CLANG_PATH: /opt/maca-3.2.1/mxgpu_llvm/bin\
    LD_LIBRARY_PATH: /opt/maca-3.2.1/lib:/opt/maca-3.2.1/mxgpu_llvm/lib:/opt/mxdriver/lib:/opt/maca-3.2.1/ompi/lib:/opt/maca-3.2.1/ucx/lib\
    PATH: /opt/conda/bin:/opt/conda/condabin:/opt/maca-3.2.1/tools/cu-bridge:/opt/maca-3.2.1/bin:/opt/maca-3.2.1/mxgpu_llvm/bin:/opt/maca-3.2.1/ompi/bin:/opt/maca-3.2.1/ucx/bin:/opt/mxdriver/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\
    HF_HUB_OFFLINE: \"1\"\
    TRANSFORMERS_OFFLINE: \"1\"' \
  examples/gpt2/conf/train_single.yaml

sed -i 's|tensor_model_parallel_size: 2|tensor_model_parallel_size: 1|' \
  examples/gpt2/conf/train/small.yaml

sed -i 's|pipeline_model_parallel_size: 8|pipeline_model_parallel_size: 1|' \
  examples/gpt2/conf/train/small.yaml

sed -i 's|use_flash_attn: True|use_flash_attn: False|' \
  examples/gpt2/conf/train/small.yaml

sed -i 's|train_iters: 20|train_iters: 5|' \
  examples/gpt2/conf/train/small.yaml

sed -i 's|lr_warmup_iters: 500|lr_warmup_iters: 1|' \
  examples/gpt2/conf/train/small.yaml

sed -i 's|data_path: ${data_path:./data/pile_wikipedia_demo}|data_path: /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy/data/pile_wikipedia_demo|' \
  examples/gpt2/conf/train/small.yaml
```

此外，GPT2 tokenizer 配置最终应包含：

```yaml
tokenizer:
  legacy_tokenizer: true
  tokenizer_type: GPT2BPETokenizer
  vocab_file: ./examples/gpt2/tokenizer/vocab.json
  merge_file: ./examples/gpt2/tokenizer/merges.txt
```

#### 3.4.3 GPT2 路线暴露出的主要问题

GPT2 这条线，依次暴露了以下问题：

1. `before_start` 激活了错误的 conda 环境
   - 原来是 `base`
   - 正确应为 `flagscale-train`

2. 子进程没有拿到 MACA 相关环境变量
   - 没有 `MACA_HOME` 等变量时，Triton / TransformerEngine 的导入和初始化会不稳定

3. 配置里用了当前仓库不支持的 OmegaConf resolver
   - 问题项是：

```yaml
data_path: ${data_path:./data/pile_wikipedia_demo}
```

   - 直接报：

```text
omegaconf.errors.UnsupportedInterpolationType: Unsupported interpolation type data_path
```

4. `flagscale/train/train.py` 与 Metax 训练参数补丁不一致
   - 代码里尝试访问 `te_fl_prefer`、`enable_flag_gems` 之类字段
   - 而 Metax 路径下实际注册的参数名并不完全对应
   - 因此远端做了一个临时热修，用 `getattr(...)` 兜底

5. 学习率 warmup 步数与训练总步数不匹配
   - 因为把 `train_iters` 大幅减小了，但 `lr_warmup_iters` 仍然很大
   - 正确处理是把 `lr_warmup_iters` 改到 `1`

6. 仍然会重新触发 Hugging Face 下载
   - 说明数据 / tokenizer 初始化并没有完全走本地 legacy 路径
   - 修正方式：
     - `legacy_tokenizer: true`
     - `HF_HUB_OFFLINE=1`
     - `TRANSFORMERS_OFFLINE=1`

7. 最终在真正运行期触发 C550 非法访存
   - 当 GPT2 路线终于跑到模型执行阶段后，最终炸在 embedding / position embedding 相关路径
   - 报错位置大致在：
     - `position_embeddings(position_ids)`
     - `torch.embedding`
     - MetaX runtime illegal address / Xnack fault

#### 3.4.4 为什么后来放弃 GPT2

GPT2 对排查基础问题很有帮助，但不适合作为最终 smoke test 路线，原因有三点：

- 仓库里的 demo 数据路径其实更匹配 Aquila
- GPT2 这条路是额外迁就出来的，不是这套仓库最自然的示例链路
- 最终的运行时崩溃正好落在 position embedding 路径，而 Aquila 使用的是另一套更贴近本仓库验证路径的配置

### 3.5 第五阶段：转向 Aquila，作为主验证路径

后续之所以切到 Aquila，是因为仓库结构和 README 已经指向它才是更适合的 demo/训练路径：

- `pile_wikipedia_demo` 本身就与 Aquila 示例配套
- `examples/aquila/tokenizer/` 在仓库中现成存在
- Aquila 的配置天然是：
  - `legacy_tokenizer: true`
  - `AquilaTokenizerFS`
  - rotary position embedding
  - `no_position_embedding: true`

这比 GPT2 更像是仓库原作者本来想验证的主线路。

#### 3.5.1 最初的 Aquila 7B smoke test 配置缩减

最开始对 Aquila 做了如下修改：

```bash
cd /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy

cp examples/aquila/conf/train/7b.yaml examples/aquila/conf/train/7b.yaml.bak

sed -i 's|use_flash_attn: True|use_flash_attn: False|' examples/aquila/conf/train/7b.yaml
sed -i 's|train_samples: 1002539063|train_samples: 80|' examples/aquila/conf/train/7b.yaml
sed -i 's|micro_batch_size: 2|micro_batch_size: 1|' examples/aquila/conf/train/7b.yaml
sed -i 's|global_batch_size: 1728|global_batch_size: 8|' examples/aquila/conf/train/7b.yaml
sed -i 's|lr_warmup_samples: 3076172|lr_warmup_samples: 1|' examples/aquila/conf/train/7b.yaml
sed -i 's|data_path: ${data_path:??}|data_path: /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy/data/pile_wikipedia_demo|' \
  examples/aquila/conf/train/7b.yaml

grep -n 'data_path\|train_samples\|micro_batch_size\|global_batch_size\|lr_warmup_samples\|use_flash_attn' \
  examples/aquila/conf/train/7b.yaml
```

#### 3.5.2 Aquila 首次失败：launcher 子进程没有继承完整运行环境

一开始的 `examples/aquila/conf/train.yaml` 只带了：

- `CUDA_VISIBLE_DEVICES`
- `CUDA_DEVICE_MAX_CONNECTIONS`

这导致子进程里 Triton Metax backend 初始化时，`MACA_HOME` 等变量依然缺失，最终掉到 `NoneType` 报错。

最终修正命令如下：

```bash
cd /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy

grep -q 'MACA_HOME:' examples/aquila/conf/train.yaml || \
sed -i '/CUDA_DEVICE_MAX_CONNECTIONS: 1/a\
    MACA_HOME: /opt/maca-3.2.1\
    MACA_PATH: /opt/maca-3.2.1\
    CUCC_PATH: /opt/maca-3.2.1/tools/cu-bridge\
    CUDA_PATH: /opt/maca-3.2.1/tools/cu-bridge\
    DEVINFO_ROOT: /opt/maca-3.2.1\
    MACA_CLANG: /opt/maca-3.2.1/mxgpu_llvm\
    MACA_CLANG_PATH: /opt/maca-3.2.1/mxgpu_llvm/bin\
    LD_LIBRARY_PATH: /opt/maca-3.2.1/lib:/opt/maca-3.2.1/mxgpu_llvm/lib:/opt/mxdriver/lib:/opt/maca-3.2.1/ompi/lib:/opt/maca-3.2.1/ucx/lib\
    PATH: /opt/conda/bin:/opt/conda/condabin:/opt/maca-3.2.1/tools/cu-bridge:/opt/maca-3.2.1/bin:/opt/maca-3.2.1/mxgpu_llvm/bin:/opt/maca-3.2.1/ompi/bin:/opt/maca-3.2.1/ucx/bin:/opt/mxdriver/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\
    HF_HUB_OFFLINE: \"1\"\
    TRANSFORMERS_OFFLINE: \"1\"' \
  examples/aquila/conf/train.yaml

grep -q 'before_start:' examples/aquila/conf/train.yaml || \
sed -i '/entrypoint: \.\/flagscale\/train\/train_gpt.py/a\
  cmds:\
    before_start: source /opt/conda/bin/activate flagscale-train' \
  examples/aquila/conf/train.yaml
```

校验：

```bash
sed -n '1,80p' examples/aquila/conf/train.yaml
```

#### 3.5.3 Aquila 接着遇到 TransformerEngine attention backend 接口不兼容

环境传播修好后，训练往前推进到了 attention backend 选择阶段，然后报：

```text
AttributeError: module 'transformer_engine_torch' has no attribute 'get_fused_attn_backend'
```

这一步的结论是：

- 当前服务器上安装的 `transformer_engine` 包，与这条 legacy 代码路径期望的 API 不一致
- 问题出在 fused attention backend 选择，不是整个训练栈从根上坏掉

为了绕开这一层，最终使用了 `unfused` attention：

```bash
sed -i '/use_flash_attn: False/a\
  attention_backend: unfused' examples/aquila/conf/train/7b.yaml
```

#### 3.5.4 切成 unfused 之后，又遇到原始 7B 配置 OOM

把 attention backend 改成 `unfused` 之后，之前的 TE API 不兼容报错消失了，但原始 Aquila 7B 配置在 C550 上依然显存过大，出现 OOM。

因此后续不再试图“直接跑原始 7B”，而是进一步压缩成真正可执行的 mini smoke 配置。

#### 3.5.5 最终生效的 Aquila mini smoke 配置缩减

最终使配置足够小、可以稳定运行的命令如下：

```bash
sed -i 's|seq_length: 2048|seq_length: 512|' examples/aquila/conf/train/7b.yaml
sed -i 's|max_position_embeddings: 2048|max_position_embeddings: 512|' examples/aquila/conf/train/7b.yaml
sed -i 's|num_layers: 32|num_layers: 8|' examples/aquila/conf/train/7b.yaml
sed -i 's|hidden_size: 4096|hidden_size: 1024|' examples/aquila/conf/train/7b.yaml
sed -i 's|num_attention_heads: 32|num_attention_heads: 16|' examples/aquila/conf/train/7b.yaml
sed -i 's|multiple_of: 256|multiple_of: 128|' examples/aquila/conf/train/7b.yaml
```

同时，还补了显存分配器配置：

```bash
grep -q 'PYTORCH_CUDA_ALLOC_CONF:' examples/aquila/conf/train.yaml || \
sed -i '/TRANSFORMERS_OFFLINE: "1"/a\
    PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True' examples/aquila/conf/train.yaml
```

校验命令：

```bash
grep -n 'seq_length\|max_position_embeddings\|num_layers\|hidden_size\|num_attention_heads\|multiple_of' \
  examples/aquila/conf/train/7b.yaml
```

最终实际工作的 mini smoke 配置核心参数是：

- `num_layers: 8`
- `hidden_size: 1024`
- `num_attention_heads: 16`
- `seq_length: 512`
- `max_position_embeddings: 512`
- `multiple_of: 128`
- `micro_batch_size: 1`
- `global_batch_size: 8`
- `train_samples: 16`
- `use_flash_attn: false`
- `attention_backend: unfused`

#### 3.5.6 接下来暴露的问题：自动恢复旧 checkpoint 导致不兼容

当 mini 配置第一次跑起来时，新的问题已经不再是 OOM，也不是算子崩溃，而是 checkpoint 恢复失败。

根因是：

- runner 默认会把 `train.system.checkpoint.load` 指向 `${experiment.exp_dir}/checkpoints`
- 而之前这个目录里残留了旧的、不兼容的 checkpoint
- 当前 mini 模型尝试恢复旧 optimizer state，就触发了分布式 checkpoint state mismatch，例如：

```text
KeyError: optimizer.distributed...bucket_idx_0.exp_avg from model not in state dict
```

所以这一步的正确处理方式不是继续改模型，而是：

- 使用全新的 `experiment.exp_dir`
- 显式把 `train.system.checkpoint.load` 指到一个不存在的路径，强制冷启动

### 3.6 第六阶段：最终成功的冷启动 smoke test

最终成功跑通的命令如下：

```bash
cd /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy

TS=$(date +%Y%m%d_%H%M%S)

python run.py --config-path ./examples/aquila/conf --config-name train action=test \
  experiment.exp_dir=/workspace/exp/aquila7b_smoke_${TS} \
  train.system.checkpoint.load=/workspace/exp/__no_ckpt__/does_not_exist \
  train.system.checkpoint.save=/workspace/exp/aquila7b_smoke_${TS}/checkpoints
```

实际成功的输出目录是：

```bash
/workspace/exp/aquila7b_smoke_20260326_071557
```

日志中的关键成功信号包括：

先明确说明不会加载旧 checkpoint：

```text
WARNING: could not find the metadata file /workspace/exp/__no_ckpt__/does_not_exist/latest_checkpointed_iteration.txt
    will not load any checkpoints and will start from random
```

然后计算出本次训练总步数：

```text
setting training iterations to 2
```

接着真实执行训练：

```text
iteration        1/       2
iteration        2/       2
```

最后成功保存 checkpoint：

```text
successfully saved checkpoint from iteration       2 to /workspace/exp/aquila7b_smoke_20260326_071557/checkpoints
```

### 3.7 最终运行特征

在这次成功的 mini smoke test 中，观察到：

- 总参数量约 `0.31B`
- 理论内存开销大约：

```text
weight and optimizer = 2201.50 MB
activation = 336.00 MB
total = 2537.50 MB
```

- rank0 在第 1 个 iteration 后的实际显存：

```text
allocated: 2289.95556640625 MB
max allocated: 2565.23681640625 MB
reserved: 3020.0 MB
max reserved: 3020.0 MB
```

这说明当前 mini 配置已经非常安全地落在可运行区间内。

## 4. 命令索引

这一节把本次排查中最重要的命令按用途集中整理一遍。

### 4.1 构建 Metax 训练树

```bash
python tools/patch/unpatch.py --backend Megatron-LM FlagScale --task train --device-type Metax_C550
```

### 4.2 Aquila 初始 smoke 配置压缩

```bash
cd /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy

cp examples/aquila/conf/train/7b.yaml examples/aquila/conf/train/7b.yaml.bak

sed -i 's|use_flash_attn: True|use_flash_attn: False|' examples/aquila/conf/train/7b.yaml
sed -i 's|train_samples: 1002539063|train_samples: 80|' examples/aquila/conf/train/7b.yaml
sed -i 's|micro_batch_size: 2|micro_batch_size: 1|' examples/aquila/conf/train/7b.yaml
sed -i 's|global_batch_size: 1728|global_batch_size: 8|' examples/aquila/conf/train/7b.yaml
sed -i 's|lr_warmup_samples: 3076172|lr_warmup_samples: 1|' examples/aquila/conf/train/7b.yaml
sed -i 's|data_path: ${data_path:??}|data_path: /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy/data/pile_wikipedia_demo|' \
  examples/aquila/conf/train/7b.yaml
```

### 4.3 为 Aquila launcher 注入完整环境

```bash
grep -q 'MACA_HOME:' examples/aquila/conf/train.yaml || \
sed -i '/CUDA_DEVICE_MAX_CONNECTIONS: 1/a\
    MACA_HOME: /opt/maca-3.2.1\
    MACA_PATH: /opt/maca-3.2.1\
    CUCC_PATH: /opt/maca-3.2.1/tools/cu-bridge\
    CUDA_PATH: /opt/maca-3.2.1/tools/cu-bridge\
    DEVINFO_ROOT: /opt/maca-3.2.1\
    MACA_CLANG: /opt/maca-3.2.1/mxgpu_llvm\
    MACA_CLANG_PATH: /opt/maca-3.2.1/mxgpu_llvm/bin\
    LD_LIBRARY_PATH: /opt/maca-3.2.1/lib:/opt/maca-3.2.1/mxgpu_llvm/lib:/opt/mxdriver/lib:/opt/maca-3.2.1/ompi/lib:/opt/maca-3.2.1/ucx/lib\
    PATH: /opt/conda/bin:/opt/conda/condabin:/opt/maca-3.2.1/tools/cu-bridge:/opt/maca-3.2.1/bin:/opt/maca-3.2.1/mxgpu_llvm/bin:/opt/maca-3.2.1/ompi/bin:/opt/maca-3.2.1/ucx/bin:/opt/mxdriver/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\
    HF_HUB_OFFLINE: \"1\"\
    TRANSFORMERS_OFFLINE: \"1\"' \
  examples/aquila/conf/train.yaml

grep -q 'before_start:' examples/aquila/conf/train.yaml || \
sed -i '/entrypoint: \.\/flagscale\/train\/train_gpt.py/a\
  cmds:\
    before_start: source /opt/conda/bin/activate flagscale-train' \
  examples/aquila/conf/train.yaml
```

### 4.4 强制走 unfused attention

```bash
sed -i '/use_flash_attn: False/a\
  attention_backend: unfused' examples/aquila/conf/train/7b.yaml
```

### 4.5 压缩到最终可运行的 mini smoke 模型

```bash
sed -i 's|seq_length: 2048|seq_length: 512|' examples/aquila/conf/train/7b.yaml
sed -i 's|max_position_embeddings: 2048|max_position_embeddings: 512|' examples/aquila/conf/train/7b.yaml
sed -i 's|num_layers: 32|num_layers: 8|' examples/aquila/conf/train/7b.yaml
sed -i 's|hidden_size: 4096|hidden_size: 1024|' examples/aquila/conf/train/7b.yaml
sed -i 's|num_attention_heads: 32|num_attention_heads: 16|' examples/aquila/conf/train/7b.yaml
sed -i 's|multiple_of: 256|multiple_of: 128|' examples/aquila/conf/train/7b.yaml

grep -q 'PYTORCH_CUDA_ALLOC_CONF:' examples/aquila/conf/train.yaml || \
sed -i '/TRANSFORMERS_OFFLINE: "1"/a\
    PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True' examples/aquila/conf/train.yaml
```

### 4.6 最终成功的冷启动运行命令

```bash
cd /workspace/muxi-flagscale-legacy/build/Metax_C550/muxi-flagscale-legacy

TS=$(date +%Y%m%d_%H%M%S)

python run.py --config-path ./examples/aquila/conf --config-name train action=test \
  experiment.exp_dir=/workspace/exp/aquila7b_smoke_${TS} \
  train.system.checkpoint.load=/workspace/exp/__no_ckpt__/does_not_exist \
  train.system.checkpoint.save=/workspace/exp/aquila7b_smoke_${TS}/checkpoints
```

## 5. 最终结论

### 5.1 已经证明的事情

下面这些结论已经在远端 MetaX C550 服务器上被实际验证：

- `main-legacy` 的 Metax C550 训练树可以成功 build 出来
- 8 卡分布式训练任务可以正常启动
- tokenizer 和 dataset 初始化可以正常完成
- 模型构建和 distributed optimizer 初始化可以正常完成
- 训练循环可以实际执行 forward、backward、optimizer step
- checkpoint 保存链路可以正常工作
- 在以下条件下，训练闭环是可复现可运行的：
  - 使用 Aquila 路线
  - 使用 mini 模型规模
  - `attention_backend: unfused`
  - 使用冷启动 checkpoint 路径

### 5.2 还没有证明的事情

- 原始未压缩的 Aquila 7B 配置可以在当前栈上稳定训练
- 当前安装的 TransformerEngine fused attention 路线与 legacy 代码完全兼容
- GPT2 在这套数据 / tokenizer / 运行时组合下是稳定可用的 smoke test 路线

### 5.3 当前仍存在但不是 blocker 的问题

以下问题仍然存在，但已经不影响“最小训练闭环已打通”的结论：

- TransformerEngine fused attention API 不兼容：

```text
transformer_engine_torch has no attribute get_fused_attn_backend
```

- 运行中仍有一些 warning，例如：
  - `fused_indices_to_multihot has reached end of life`
  - `destroy_process_group() was not called before program exit`
  - TensorBoard / one_logger 缺失提示

这些 warning 在最终成功的 smoke test 中都没有阻止训练和保存 checkpoint。

目前结论是，
`FlagScale main-legacy + Metax_C550 的训练链路并没有从根上坏掉；基于 Aquila 的压缩版 mini smoke 配置，已经在 8 张 C550 上完成了 2 个 iteration 的冷启动训练并成功保存 checkpoint。`
