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
- 📊 强大的 Wandb 实验追踪与可视化
- 🔄 支持流式推理输出
- 💾 支持梯度累积和混合精度训练
- 📝 支持最佳模型保存和检查点管理
- 🔁 支持定期保存和自动清理检查点

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
# 预训练模式 (Pretrain)

## CPU训练（测试用）
python pretrain_sft_lora.py \
    --mode pretrain \
    --device cpu \
    --batch_size 2 \
    --epochs 1 \
    --dim 128 \
    --n_layers 2 \
    --max_seq_len 128 \
    --n_heads 4

## GPU训练（单卡）- 基础配置
python pretrain_sft_lora.py \
    --mode pretrain \
    --batch_size 32 \
    --epochs 1 \
    --learning_rate 5e-4 \
    --dim 512 \
    --n_layers 8 \
    --accumulation_steps 8 \
    --save_interval 1000 \
    --save_best_only \
    --save_last \
    --save_interval_steps 1000 \
    --keep_checkpoint_max 5

## GPU训练（单卡）- 启用Wandb监控
python pretrain_sft_lora.py \
    --mode pretrain \
    --batch_size 32 \
    --epochs 1 \
    --learning_rate 5e-4 \
    --dim 512 \
    --n_layers 8 \
    --accumulation_steps 8 \
    --use_wandb \
    --wandb_project "Zer02LLM" \
    --wandb_log_model \
    --wandb_log_code \
    --wandb_watch_model \
    --save_best_only \
    --save_last

## GPU训练（多卡）
torchrun --nproc_per_node=2 pretrain_sft_lora.py \
    --mode pretrain \
    --ddp \
    --batch_size 16 \
    --accumulation_steps 8 \
    --save_interval 1000 \
    --save_best_only \
    --save_last

# 监督微调模式 (SFT)

## 单卡训练
python pretrain_sft_lora.py \
    --mode sft \
    --batch_size 16 \
    --epochs 1 \
    --learning_rate 5e-6 \
    --dim 512 \
    --n_layers 8 \
    --max_seq_len 512 \
    --accumulation_steps 8 \
    --save_interval 1000 \
    --save_best_only \
    --save_last

## 多卡训练
torchrun --nproc_per_node=2 pretrain_sft_lora.py \
    --mode sft \
    --ddp \
    --batch_size 8 \
    --epochs 1 \
    --learning_rate 5e-6 \
    --save_best_only \
    --save_last

## 启用MoE的SFT训练
python pretrain_sft_lora.py \
    --mode sft \
    --use_moe \
    --n_routed_experts 8 \
    --num_experts_per_tok 2 \
    --batch_size 8 \
    --save_best_only \
    --save_last

# 人类偏好对齐模式 (DPO)

## 单卡训练
python pretrain_sft_lora.py \
    --mode dpo \
    --batch_size 8 \
    --epochs 2 \
    --learning_rate 1e-8 \
    --max_seq_len 3000 \
    --accumulation_steps 8 \
    --save_interval 1000 \
    --save_best_only \
    --save_last

## 多卡训练
torchrun --nproc_per_node=2 pretrain_sft_lora.py \
    --mode dpo \
    --ddp \
    --batch_size 8 \
    --epochs 2 \
    --learning_rate 1e-8 \
    --max_seq_len 3000 \
    --save_best_only \
    --save_last

## 启用MoE的DPO训练
python pretrain_sft_lora.py \
    --mode dpo \
    --use_moe \
    --batch_size 8 \
    --epochs 2 \
    --learning_rate 1e-8 \
    --max_seq_len 3000 \
    --save_best_only \
    --save_last

## 显存优化配置的DPO训练
python pretrain_sft_lora.py \
    --mode dpo \
    --batch_size 4 \
    --accumulation_steps 4 \
    --learning_rate 1e-8 \
    --max_seq_len 3000 \
    --save_best_only \
    --save_last
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
- **人类偏好对齐 (DPO)**: 基于人类偏好数据进行模型对齐训练

### 2. 模型特性
- **注意力机制**: 支持标准注意力和 Flash Attention
- **位置编码**: 使用 RoPE 进行位置信息编码
- **MoE结构**: 支持动态路由和专家选择
- **混合精度**: 支持 FP16/BF16 训练

### 3. 优化特性
- **梯度累积**: 支持大批量训练
- **分布式训练**: 支持多GPU训练
- **性能监控**: 支持 Wandb 实验追踪
  - 超参数记录：学习率、优化器配置、批大小等
  - 训练指标：损失、梯度范数、困惑度(perplexity)
  - GPU资源监控：内存占用、GPU利用率
  - 训练速度：每秒处理token数、训练时间估计
  - 模型权重：自动记录检查点文件
  - 代码与配置：可选记录代码文件和完整配置
- **流式生成**: 支持流式文本生成
- **检查点管理**: 支持最佳模型保存和自动清理

### 4. 检查点保存策略
- **定期保存**: 按步数间隔保存模型
- **最佳模型**: 保存训练过程中loss最低的模型
- **最终模型**: 保存训练结束时的模型状态
- **自动清理**: 自动删除旧的检查点以节省空间
- **数量控制**: 可配置保留的最大检查点数量

### 5. Wandb监控
- 使用`--use_wandb`启用Wandb监控
- 使用`--wandb_log_model`记录模型权重
- 使用`--wandb_log_code`记录代码文件
- 调整`--wandb_log_freq`控制记录频率
- 使用`--wandb_watch_model`监控模型梯度

### 6. 超参数搜索
- 使用Weights & Biases Sweeps进行自动化超参数优化
- 支持贝叶斯优化、网格搜索和随机搜索
- 自动早停机制，节省计算资源
- 可视化超参数重要性和相关性
- 支持不同训练模式的专用配置

```bash
# 启动预训练模式的超参数搜索（运行10次实验）
python run_sweep.py --config sweep_config.yaml --count 10 --mode pretrain

# 启动DPO模式的超参数搜索（运行5次实验）
python run_sweep.py --config sweep_config_dpo.yaml --count 5 --mode dpo

# 在特定GPU上运行超参数搜索
python run_sweep.py --config sweep_config.yaml --count 3 --gpu 0,1
```

### 7. 自动化工作流

Zer02LLM 提供了一个完整的自动化工作流脚本 `workflow.py`，可以帮助你自动化执行从设置、超参数搜索、训练、评估到部署的全流程。

#### 工作流阶段

工作流包含以下阶段：
- **设置 (setup)**: 初始化工作流环境和配置
- **超参数搜索 (sweep)**: 使用 Weights & Biases 进行超参数搜索
- **训练 (train)**: 使用最佳超参数配置训练模型
- **评估 (evaluate)**: 评估训练好的模型性能
- **分析 (analyze)**: 分析训练和评估结果
- **部署 (deploy)**: 部署最终模型

#### 一键测试命令

```bash
# 设置阶段 - 初始化工作流环境和配置
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage setup

# 超参数搜索阶段 - 运行 5 次实验
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage sweep --sweep_count 5

# 训练阶段 - 使用最佳超参数配置训练模型
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage train

# 评估阶段 - 评估训练好的模型
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage evaluate

# 分析阶段 - 分析训练和评估结果
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage analyze

# 部署阶段 - 部署最终模型
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage deploy

# 运行完整工作流（从设置到部署）
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --all
```

#### 工作流参数说明

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| mode | 训练模式 | pretrain | pretrain/sft/dpo |
| project | Wandb项目名称 | Zer02LLM_Workflow | 字符串 |
| entity | Wandb实体名称 | None | 字符串/None |
| config | 超参数搜索配置文件路径 | None | 有效文件路径 |
| output_dir | 工作流输出目录 | workflow_output | 有效目录路径 |
| sweep_count | 超参数搜索运行次数 | 5 | 正整数 |
| gpu | 指定使用的GPU | None | 如 '0,1' |
| stage | 要运行的工作流阶段 | setup | setup/sweep/train/evaluate/analyze/deploy |
| all | 运行所有工作流阶段 | False | True/False |
| wandb_host | wandb主机地址 | None | 字符串 |
| wandb_base_url | wandb基础URL | None | 字符串 |

#### 工作流输出

工作流会在指定的输出目录中生成以下文件和目录：
- `configs/`: 超参数配置文件
- `workflow_config.json`: 工作流配置和状态
- `best_config.yaml`: 最佳超参数配置
- `analysis/`: 分析结果和图表
- `eval_results_*.txt`: 评估结果

#### Windows 系统注意事项

在 Windows 系统上运行工作流脚本时，可能会遇到编码问题。如果遇到类似 `UnicodeDecodeError: 'gbk' codec can't decode byte...` 的错误，请确保所有文件读写操作都使用 UTF-8 编码：

```python
# 正确的文件读写方式
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
```

#### 完整测试流程示例

以下是在本地进行完整测试的推荐步骤：

1. **准备环境**：确保已安装所有依赖并登录到 Weights & Biases
   ```bash
   pip install -r requirements.txt
   wandb login
   ```

2. **设置阶段**：初始化工作流配置
   ```bash
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage setup
   ```

3. **超参数搜索**：运行少量实验以测试功能
   ```bash
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage sweep --sweep_count 1
   ```

4. **训练阶段**：使用最佳配置训练模型
   ```bash
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage train
   ```

5. **评估和分析**：评估模型并分析结果
   ```bash
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage evaluate
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage analyze
   ```

6. **部署阶段**：部署最终模型
   ```bash
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage deploy
   ```

在测试过程中，即使某些阶段因缺少数据或模型而无法完成，工作流脚本也会优雅地处理这些情况，并允许您继续测试后续阶段。

## 参数说明

### 模型参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| dim | 隐藏层维度 | 512 |
| n_layers | 层数 | 8 |
| n_heads | 注意力头数 | 8 |
| max_seq_len | 最大序列长度 | 2048 |
| use_moe | 是否启用 MoE | False |

### 训练参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| learning_rate | 学习率 | 5e-4 |
| batch_size | 批次大小 | 32 |
| accumulation_steps | 梯度累积步数 | 8 |
| epochs | 训练轮数 | 1 |

### DPO训练参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| learning_rate | DPO学习率 | 1e-8 |
| batch_size | 批次大小 | 8 |
| max_seq_len | 序列长度 | 3000 |
| epochs | 训练轮数 | 2 |

### 检查点参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| save_best_only | 是否只保存最佳模型 | True |
| save_last | 是否保存最后一个epoch的模型 | True |
| save_interval_steps | 每多少步保存一次模型 | 1000 |
| keep_checkpoint_max | 最多保存多少个检查点 | 5 |

## 开源协议

本项目采用 MIT 协议 - 详见 [LICENSE](LICENSE) 文件

## 完整参数配置说明

### 基础训练参数
| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| mode | 训练模式 | pretrain | pretrain/sft/dpo/test |
| out_dir | 输出目录 | out | 任意有效路径 |
| epochs | 训练轮数 | 1 | 正整数 |
| batch_size | 批次大小 | 32 | 正整数 |
| learning_rate | 学习率 | 5e-4 | 浮点数 |
| device | 训练设备 | cuda:0/cpu | cuda:N/cpu |
| dtype | 数据类型 | bfloat16 | float16/bfloat16/float32 |

### 日志和监控参数
| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| use_wandb | 是否使用wandb | False | True/False |
| wandb_project | wandb项目名 | Zer02LLM | 字符串 |
| wandb_run_name | wandb运行名称 | None | 字符串/None |
| wandb_log_model | 是否记录模型权重 | False | True/False |
| wandb_log_code | 是否记录代码 | False | True/False |
| wandb_log_freq | wandb记录频率 | 1 | 正整数 |
| wandb_watch_model | 是否监控模型梯度 | False | True/False |
| wandb_watch_log | wandb.watch的log参数 | gradients | gradients/all/None |
| wandb_watch_log_freq | wandb.watch的log_freq | 100 | 正整数 |
| log_interval | 日志记录间隔 | 100 | 正整数 |
| save_interval | 保存间隔 | 100 | 正整数 |

### 检查点保存参数
| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| save_best_only | 是否只保存最佳模型 | True | True/False |
| save_last | 是否保存最后一个epoch的模型 | True | True/False |
| save_interval_steps | 每多少步保存一次模型 | 1000 | 正整数 |
| keep_checkpoint_max | 最多保存多少个检查点 | 5 | 正整数 |

### 分布式训练参数
| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| num_workers | 数据加载线程数 | 1 | 正整数 |
| ddp | 是否使用DDP | False | True/False |
| local_rank | 本地进程序号 | -1 | 整数 |

### 优化器参数
| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| accumulation_steps | 梯度累积步数 | 8 | 正整数 |
| grad_clip | 梯度裁剪值 | 1.0 | 正浮点数 |
| warmup_iters | 预热迭代次数 | 0 | 非负整数 |

### 模型结构参数
| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| dim | 隐藏层维度 | 512 | 正整数 |
| n_layers | 层数 | 8 | 正整数 |
| max_seq_len | 最大序列长度 | 512 | 正整数 |
| vocab_size | 词表大小 | 4000 | 正整数 |
| n_heads | 注意力头数 | 8 | 正整数 |
| n_kv_heads | KV注意力头数 | None | 正整数/None |
| rope_theta | RoPE角度参数 | 10000.0 | 正浮点数 |
| dropout | Dropout比率 | 0.1 | 0~1浮点数 |
| norm_eps | 归一化epsilon | 1e-8 | 正浮点数 |
| multiple_of | 维度倍数 | 64 | 正整数 |
| flash_attn | 是否使用Flash Attention | False | True/False |

### MoE相关参数
| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| use_moe | 是否启用MoE | False | True/False |
| n_routed_experts | 路由专家数量 | 8 | 正整数 |
| num_experts_per_tok | 每个token的专家数 | 2 | 正整数 |
| n_shared_experts | 共享专家数量 | 0 | 非负整数 |
| scoring_func | 评分函数 | softmax | softmax |
| aux_loss_alpha | 辅助损失权重 | 0.1 | 正浮点数 |
| seq_aux | 是否使用序列辅助损失 | True | True/False |
| norm_topk_prob | 是否归一化topk概率 | True | True/False |

### DPO相关参数
| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| dpo_beta | DPO loss的beta参数 | 0.1 | 正浮点数 |
| ref_model_path | 参考模型路径 | None | 字符串/None |

### 数据相关参数
| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| data_path | 训练数据路径 | ./dataset/pretrain_hq.jsonl | 有效文件路径 |

## 显存优化配置

```bash
# 16GB显存配置
python pretrain_sft_lora.py --mode pretrain --dim 512 --n_layers 8 --batch_size 16 --accumulation_steps 16

# 24GB显存配置
python pretrain_sft_lora.py --mode pretrain --dim 768 --n_layers 12 --batch_size 24 --accumulation_steps 8

# 40GB显存配置
python pretrain_sft_lora.py --mode pretrain --dim 1024 --n_layers 16 --batch_size 32 --accumulation_steps 4

# DPO训练 16GB显存配置
python pretrain_sft_lora.py --mode dpo --dim 512 --n_layers 8 --batch_size 4 --accumulation_steps 8 --max_seq_len 2048

# DPO训练 24GB显存配置
python pretrain_sft_lora.py --mode dpo --dim 768 --n_layers 12 --batch_size 6 --accumulation_steps 6 --max_seq_len 2048

# DPO训练 40GB显存配置
python pretrain_sft_lora.py --mode dpo --dim 1024 --n_layers 16 --batch_size 8 --accumulation_steps 4 --max_seq_len 3000
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

### 4. 检查点管理
- 合理设置保存间隔
- 控制检查点数量
- 及时清理旧检查点
- 保存最佳模型

### 5. Wandb监控
- 使用`--use_wandb`启用Wandb监控
- 使用`--wandb_log_model`记录模型权重
- 使用`--wandb_log_code`记录代码文件
- 调整`--wandb_log_freq`控制记录频率
- 使用`--wandb_watch_model`监控模型梯度

### 6. 超参数搜索
- 使用Weights & Biases Sweeps进行自动化超参数优化
- 支持贝叶斯优化、网格搜索和随机搜索
- 自动早停机制，节省计算资源
- 可视化超参数重要性和相关性
- 支持不同训练模式的专用配置

```bash
# 启动预训练模式的超参数搜索（运行10次实验）
python run_sweep.py --config sweep_config.yaml --count 10 --mode pretrain

# 启动DPO模式的超参数搜索（运行5次实验）
python run_sweep.py --config sweep_config_dpo.yaml --count 5 --mode dpo

# 在特定GPU上运行超参数搜索
python run_sweep.py --config sweep_config.yaml --count 3 --gpu 0,1
```

### 7. 自动化工作流

Zer02LLM 提供了一个完整的自动化工作流脚本 `workflow.py`，可以帮助你自动化执行从设置、超参数搜索、训练、评估到部署的全流程。

#### 工作流阶段

工作流包含以下阶段：
- **设置 (setup)**: 初始化工作流环境和配置
- **超参数搜索 (sweep)**: 使用 Weights & Biases 进行超参数搜索
- **训练 (train)**: 使用最佳超参数配置训练模型
- **评估 (evaluate)**: 评估训练好的模型性能
- **分析 (analyze)**: 分析训练和评估结果
- **部署 (deploy)**: 部署最终模型

#### 一键测试命令

```bash
# 设置阶段 - 初始化工作流环境和配置
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage setup

# 超参数搜索阶段 - 运行 5 次实验
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage sweep --sweep_count 5

# 训练阶段 - 使用最佳超参数配置训练模型
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage train

# 评估阶段 - 评估训练好的模型
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage evaluate

# 分析阶段 - 分析训练和评估结果
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage analyze

# 部署阶段 - 部署最终模型
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --stage deploy

# 运行完整工作流（从设置到部署）
python workflow.py --mode pretrain --project your_project_name --output_dir ./test_output --all
```

#### Windows 系统注意事项

在 Windows 系统上运行工作流脚本时，可能会遇到编码问题。如果遇到类似 `UnicodeDecodeError: 'gbk' codec can't decode byte...` 的错误，请确保所有文件读写操作都使用 UTF-8 编码：

```python
# 正确的文件读写方式
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
```

#### 完整测试流程示例

以下是在本地进行完整测试的推荐步骤：

1. **准备环境**：确保已安装所有依赖并登录到 Weights & Biases
   ```bash
   pip install -r requirements.txt
   wandb login
   ```

2. **设置阶段**：初始化工作流配置
   ```bash
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage setup
   ```

3. **超参数搜索**：运行少量实验以测试功能
   ```bash
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage sweep --sweep_count 1
   ```

4. **训练阶段**：使用最佳配置训练模型
   ```bash
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage train
   ```

5. **评估和分析**：评估模型并分析结果
   ```bash
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage evaluate
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage analyze
   ```

6. **部署阶段**：部署最终模型
   ```bash
   python workflow.py --mode pretrain --project test_project --output_dir ./test_output --stage deploy
   ```

在测试过程中，即使某些阶段因缺少数据或模型而无法完成，工作流脚本也会优雅地处理这些情况，并允许您继续测试后续阶段。

