# Zer02LLM

一个轻量级的大语言模型(LLM)训练和推理框架，专注于高效实现和性能优化。

## 项目特点

- 🚀 高效的 Transformer 架构实现
- 🎯 支持 MoE (Mixture of Experts) 结构
- 💡 实现了 Flash Attention 优化
- 🔄 支持 RoPE (Rotary Position Embedding) 位置编码
- 📦 模块化设计，易于扩展
- 🛠 内置性能优化机制
- 🔤 支持自定义 tokenizer
- 📊 支持 Wandb 实验追踪
- 🔄 支持流式推理输出
- 💾 支持梯度累积和混合精度训练

## 目录结构

```
Zer02LLM_all/
├── model/                # 模型相关代码
│   ├── model.py         # 核心模型实现
│   └── LLMconfig.py     # 模型配置类
├── tokenizer/           # tokenizer 相关文件
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
├── datasets.py          # 数据集处理
├── pretrain_sft_lora.py # 训练脚本
├── eval_model.py        # 评估脚本
├── requirements.txt     # 依赖包列表
└── README.md           # 项目说明文档
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保以下文件/目录存在：
- `./tokenizer/` - tokenizer目录（包含必要的tokenizer文件）
- `./dataset/pretrain_hq.jsonl` - 预训练数据文件
- `./out/` - 模型输出目录（会自动创建）

### 3. 模型训练

```bash
# CPU训练（测试用）
python pretrain_sft_lora.py --device cpu --mode pretrain --batch_size 2 --epochs 1 --dim 128 --n_layers 2 --max_seq_len 128 --n_heads 4

# GPU训练（单卡）
python pretrain_sft_lora.py --mode pretrain --batch_size 32 --epochs 1 --learning_rate 5e-4 --dim 512 --n_layers 8

# 分布式训练（多卡）
torchrun --nproc_per_node=2 pretrain_sft_lora.py --mode pretrain --ddp --batch_size 16
```

### 4. 模型评估

```bash
# 评估预训练模型
python eval_model.py --model_mode 0 --dim 512 --n_layers 8

# 评估SFT模型（带流式输出）
python eval_model.py --model_mode 1 --dim 512 --n_layers 8 --stream True
```

## 核心功能说明

### 1. 训练模式
- **预训练 (Pretrain)**: 从头训练模型
- **监督微调 (SFT)**: 基于预训练模型进行对话能力训练

### 2. 模型特性
- **注意力机制**: 支持标准注意力和 Flash Attention
- **位置编码**: 使用 RoPE 进行位置信息编码
- **MoE结构**: 支持动态路由和专家选择
- **混合精度**: 支持 FP16/BF16 训练

### 3. 优化特性
- **梯度累积**: 支持大批量训练
- **分布式训练**: 支持多GPU训练
- **性能监控**: 支持 Wandb 实验追踪
- **流式生成**: 支持流式文本生成

## 显存优化配置

```bash
# 16GB显存配置
python pretrain_sft_lora.py --mode pretrain --dim 512 --n_layers 8 --batch_size 16 --accumulation_steps 16

# 24GB显存配置
python pretrain_sft_lora.py --mode pretrain --dim 768 --n_layers 12 --batch_size 24 --accumulation_steps 8

# 40GB显存配置
python pretrain_sft_lora.py --mode pretrain --dim 1024 --n_layers 16 --batch_size 32 --accumulation_steps 4
```

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
- 使用混合精度训练

### 3. 训练稳定性
- 调整学习率
- 使用 warmup
- 启用梯度裁剪
- 调整 batch_size 和 accumulation_steps

## 参数说明

### 模型参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| dim | 隐藏层维度 | 512 |
| n_layers | 层数 | 8 |
| n_heads | 注意力头数 | 8 |
| max_seq_len | 最大序列长度 | 2048 |

### 训练参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| learning_rate | 学习率 | 5e-4 |
| batch_size | 批次大小 | 32 |
| accumulation_steps | 梯度累积步数 | 8 |
| epochs | 训练轮数 | 1 |

## 开源协议

本项目采用 MIT 协议 - 详见 [LICENSE](LICENSE) 文件

