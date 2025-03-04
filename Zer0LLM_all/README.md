# Zer02LLM_all

一个轻量级的 LLM (Large Language Model) 实现项目，专注于模型训练和推理优化。

## 项目特点

- 🚀 高效的 Transformer 架构实现
- 🎯 支持 MoE (Mixture of Experts) 结构
- 💡 实现了 Flash Attention 优化
- 🔄 支持 RoPE (Rotary Position Embedding) 位置编码
- 📦 模块化设计，易于扩展
- 🛠 内置性能优化机制
- 🔤 支持自定义 tokenizer

## 目录结构

```
Zer02LLM_all/
├── model/              # 模型相关代码
│   ├── model.py       # 核心模型实现
│   └── LLMconfig.py   # 模型配置类
├── tokenizer/         # tokenizer 相关文件
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
├── datasets.py        # 数据集处理
├── pretrain_sft_lora.py  # 训练脚本
└── README.md
```

## 核心功能

- **高效注意力机制**: 实现了包含 Flash Attention 在内的优化注意力计算
- **MoE 专家系统**: 支持动态路由和专家选择
- **位置编码优化**: 采用 RoPE 进行位置信息编码
- **灵活的模型配置**: 支持自定义模型参数和结构
- **流式生成**: 支持文本的流式生成输出
- **自定义 Tokenizer**: 支持使用自定义的 tokenizer 进行训练

## 环境准备

### 必要文件和目录
- `./tokenizer/` - tokenizer目录（包含必要的tokenizer文件）
- `./dataset/pretrain_hq.jsonl` - 预训练数据文件
- `./out/` - 模型输出目录（会自动创建）

### 安装依赖
```bash
pip install -r requirements.txt
```

## 训练指南

### 基础训练命令

```bash
# CPU训练（测试用）
python pretrain_sft_lora.py --device cpu --mode pretrain --batch_size 2 --epochs 1 --dim 128 --n_layers 2 --max_seq_len 128 --n_heads 4 --data_path ./datasets/pretrain_hq.jsonl

# GPU预训练
python pretrain_sft_lora.py --mode pretrain --batch_size 32 --epochs 1 --learning_rate 5e-4 --dim 512 --n_layers 8 --data_path ./datasets/pretrain_hq.jsonl

# SFT微调
python pretrain_sft_lora.py --mode sft --batch_size 32 --epochs 1 --learning_rate 5e-4 --dim 512 --n_layers 8 --data_path ./datasets/sft_data.jsonl

# MoE模型训练
python pretrain_sft_lora.py --mode pretrain --use_moe --n_routed_experts 8 --num_experts_per_tok 2 --batch_size 32 --dim 512 --n_layers 8

# 分布式训练
torchrun --nproc_per_node=2 pretrain_sft_lora.py --mode pretrain --ddp --batch_size 16
```

### 显存优化配置

```bash
# 16GB显存配置
python pretrain_sft_lora.py --mode pretrain --dim 512 --n_layers 8 --batch_size 16 --accumulation_steps 16

# 24GB显存配置
python pretrain_sft_lora.py --mode pretrain --dim 768 --n_layers 12 --batch_size 24 --accumulation_steps 8

# 40GB+显存配置
python pretrain_sft_lora.py --mode pretrain --dim 1024 --n_layers 16 --batch_size 32 --accumulation_steps 4
```

### 重要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --dim | 模型维度 | 512 |
| --n_layers | 模型层数 | 8 |
| --max_seq_len | 最大序列长度 | 512 |
| --batch_size | 批次大小 | 32 |
| --accumulation_steps | 梯度累积步数 | 8 |
| --learning_rate | 学习率 | 5e-4 |

### 训练监控
- 使用wandb监控：`--use_wandb`
- 日志间隔：`--log_interval`
- 保存间隔：`--save_interval`

### 模型保存
- 预训练模型：`out/pretrain_{dim}.pth`
- SFT模型：`out/sft_{dim}.pth`
- MoE模型：添加`_moe`后缀

## 常见问题解决

### 1. 显存不足
- 减小 batch_size
- 增加 accumulation_steps
- 减小模型维度或层数
- 使用 float16/bfloat16 精度

### 2. 训练速度优化
- 启用 Flash Attention
- 使用分布式训练
- 优化数据加载（增加 num_workers）

### 3. 训练稳定性
- 调整学习率
- 使用 warmup
- 启用梯度裁剪

## 开源协议

本项目采用 MIT 协议 - 详见 [LICENSE](LICENSE) 文件

