# Zer02LLM_all

一个轻量级的 LLM (Large Language Model) 实现项目，专注于模型训练和推理优化。

## 项目特点

- 🚀 高效的 Transformer 架构实现
- 🎯 支持 MoE (Mixture of Experts) 结构
- 💡 实现了 Flash Attention 优化
- 🔄 支持 RoPE (Rotary Position Embedding) 位置编码
- 📦 模块化设计，易于扩展
- 🛠 内置性能优化机制

## 核心功能

- **高效注意力机制**: 实现了包含 Flash Attention 在内的优化注意力计算
- **MoE 专家系统**: 支持动态路由和专家选择
- **位置编码优化**: 采用 RoPE 进行位置信息编码
- **灵活的模型配置**: 支持自定义模型参数和结构
- **流式生成**: 支持文本的流式生成输出

## 安装说明

```bash

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```python
from model.model import LLM
from model.LLMconfig import LLMconfig

# 配置模型参数
config = LLMconfig(
    vocab_size=32000,
    dim=512,
    n_layers=6,
    n_heads=8,
    max_seq_len=2048
)

# 初始化模型
model = LLM(config)

# 使用模型进行生成
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)
```

## 主要组件

- **FFN (Feed Forward Network)**: 实现了基于 SwiGLU 的前馈网络
- **Attention**: 支持多头注意力机制和 KV 缓存
- **MoEgate**: 实现了专家选择和路由机制
- **RMSnorm**: 高效的层归一化实现

## 性能优化

- 实现了 Flash Attention 机制
- 支持 KV 缓存优化推理速度
- MoE 结构提升模型容量和效率
- 优化的位置编码计算

## 开源协议

本项目采用 MIT 协议 - 详见 [LICENSE](LICENSE) 文件

## 训练指南

### 环境准备

在开始训练之前，请确保您有以下必要文件和目录：

- `./model/zer02llm_tokenizer` - tokenizer目录
- `./dataset/pretrain_hq.jsonl` - 预训练数据文件
- `./out` - 模型输出目录(会自动创建)

### 训练模式

项目支持两种主要的训练模式：

1. **预训练模式 (pretrain)**: 从头开始训练模型
2. **SFT模式 (sft)**: 在预训练模型基础上进行微调

### 基本训练命令

#### 预训练模式
```bash
python pretrain_sft_lora.py \
    --mode pretrain \
    --batch_size 32 \
    --epochs 1 \
    --learning_rate 5e-4 \
    --dim 512 \
    --n_layers 8 \
    --data_path ./dataset/pretrain_hq.jsonl
```

#### SFT微调模式
```bash
python pretrain_sft_lora.py \
    --mode sft \
    --batch_size 32 \
    --epochs 1 \
    --learning_rate 5e-4 \
    --dim 512 \
    --n_layers 8 \
    --data_path ./dataset/sft_data.jsonl
```

### 重要参数说明

- `--dim`: 模型维度，默认512
- `--n_layers`: 模型层数，默认8
- `--max_seq_len`: 最大序列长度，默认512
- `--batch_size`: 批次大小，默认32
- `--accumulation_steps`: 梯度累积步数，默认8
- `--learning_rate`: 学习率，默认5e-4

### 高级训练功能

#### MoE(混合专家)模型训练
```bash
python pretrain_sft_lora.py \
    --mode pretrain \
    --use_moe \
    --n_routed_experts 8 \
    --num_experts_per_tok 2
```

#### 分布式训练(DDP)
```bash
torchrun --nproc_per_node=2 pretrain_sft_lora.py \
    --mode pretrain \
    --ddp \
    --batch_size 16
```

### 训练监控

- 使用wandb监控：添加 `--use_wandb` 参数
- 日志间隔设置：`--log_interval`
- 模型保存间隔：`--save_interval`

### 模型保存

- 模型自动保存在 `out` 目录
- 预训练模型：`pretrain_{dim}.pth`
- SFT模型：`sft_{dim}.pth`
- MoE模型会在文件名后添加 `_moe` 后缀

### 训练建议

1. 建议从小规模开始测试：

```bash
python pretrain_sft_lora.py \
    --mode pretrain \
    --dim 128 \
    --n_layers 4 \
    --batch_size 8 \
    --epochs 1
```

2. 显存优化建议：
   - 如果显存不足，可以：
     - 减小 batch_size
     - 增加 accumulation_steps
     - 减小模型维度(dim)或层数(n_layers)

3. 数据准备：
   - 确保数据格式符合 `PretrainDataset` 或 `SFTDataset` 的要求
   - 根据GPU显存大小调整训练参数

### 常见问题

1. **显存不足**：
   - 首先尝试减小 batch_size
   - 可以通过增加 accumulation_steps 来模拟更大的 batch_size
   - 最后考虑减小模型规模

2. **训练速度慢**：
   - 检查是否启用了 Flash Attention
   - 考虑使用分布式训练
   - 优化数据加载流程（增加 num_workers）

3. **训练不稳定**：
   - 检查学习率是否合适
   - 适当调整 warmup_iters
   - 考虑使用梯度裁剪（grad_clip）

## 快速训练指南

### 1. 单卡训练

```bash
# 创建训练脚本 train.sh
cat << EOF > train.sh
#!/bin/bash

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0

# 基础训练命令
python pretrain_sft_lora.py \
    --mode pretrain \
    --batch_size 32 \
    --epochs 1 \
    --learning_rate 5e-4 \
    --dim 512 \
    --n_layers 8 \
    --data_path ./dataset/pretrain_hq.jsonl
EOF

# 添加执行权限
chmod +x train.sh

# 启动训练
./train.sh
```

### 2. 多卡分布式训练

```bash
# 创建分布式训练脚本 train_ddp.sh
cat << EOF > train_ddp.sh
#!/bin/bash

# 设置 CUDA 可见设备
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 获取 GPU 数量
NUM_GPUS=\$(echo \$CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# 启动分布式训练
torchrun --nproc_per_node=\$NUM_GPUS pretrain_sft_lora.py \
    --mode pretrain \
    --ddp \
    --batch_size 16 \
    --epochs 1 \
    --learning_rate 5e-4 \
    --dim 512 \
    --n_layers 8 \
    --data_path ./dataset/pretrain_hq.jsonl
EOF

# 添加执行权限
chmod +x train_ddp.sh

# 启动训练
./train_ddp.sh
```

### 3. 后台训练（使用 tmux）

```bash
# 安装 tmux（如果未安装）
apt-get install tmux  # Ubuntu/Debian
# 或
yum install tmux      # CentOS/RHEL

# 创建新的 tmux 会话
tmux new -s train

# 在 tmux 会话中启动训练
./train.sh  # 单卡训练
# 或
./train_ddp.sh  # 多卡训练

# 分离 tmux 会话（Ctrl+B 然后按 D）
# 重新连接会话
tmux attach -t train
```

### 4. 训练参数说明

#### 基础参数
- `--mode`: 训练模式 [pretrain/sft]
- `--batch_size`: 批次大小
- `--epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--dim`: 模型维度
- `--n_layers`: 模型层数

#### 高级参数
- `--max_seq_len`: 最大序列长度，默认512
- `--accumulation_steps`: 梯度累积步数，默认8
- `--dtype`: 数据类型 [float32/float16/bfloat16]
- `--flash_attn`: 启用 Flash Attention
- `--use_wandb`: 启用 Wandb 监控

### 5. 显存优化建议

根据显卡显存大小选择合适的配置：

#### 16GB 显存配置
```bash
python pretrain_sft_lora.py \
    --mode pretrain \
    --dim 512 \
    --n_layers 8 \
    --batch_size 16 \
    --accumulation_steps 16 \
    --max_seq_len 512
```

#### 24GB 显存配置
```bash
python pretrain_sft_lora.py \
    --mode pretrain \
    --dim 768 \
    --n_layers 12 \
    --batch_size 24 \
    --accumulation_steps 8 \
    --max_seq_len 512
```

#### 40GB+ 显存配置
```bash
python pretrain_sft_lora.py \
    --mode pretrain \
    --dim 1024 \
    --n_layers 16 \
    --batch_size 32 \
    --accumulation_steps 4 \
    --max_seq_len 1024
```

### 6. 训练监控

1. 实时查看日志：
```bash
tail -f train.log  # 如果将输出重定向到了日志文件
```

2. 查看 GPU 使用情况：
```bash
watch -n 1 nvidia-smi  # 每秒更新一次
```

3. 使用 Wandb 监控（需要先注册账号）：
```bash
# 添加 wandb 参数
python pretrain_sft_lora.py \
    --mode pretrain \
    --use_wandb \
    --wandb_project "项目名称" \
    [其他参数]
```

### 7. 常见问题处理

1. 显存不足：
   - 减小 batch_size
   - 增加 accumulation_steps
   - 减小模型维度(dim)或层数(n_layers)
   - 使用 float16/bfloat16 精度

2. 训练中断恢复：
   - 使用 tmux 可以防止 SSH 断开影响
   - 可以从最近的检查点恢复训练

3. 多卡训练问题：
   - 确保 CUDA_VISIBLE_DEVICES 设置正确
   - 检查网络连接（多机训练时）
   - 适当调整每张卡的 batch_size

