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
# 克隆项目
git clone https://github.com/yourusername/fastllm-learning.git
cd fastllm-learning

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

