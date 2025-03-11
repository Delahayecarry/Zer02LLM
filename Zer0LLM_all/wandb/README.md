# Zer02LLM Wandb工作流

本目录包含用于管理Zer02LLM模型训练完整工作流的工具，基于Weights & Biases (wandb)实现自动化的超参数搜索、模型训练、评估和分析。

## 文件说明

- `workflow.py`: 完整的工作流管理脚本，整合了超参数搜索、模型训练、评估和分析等功能
- `sweep_config.yaml`: 预训练和SFT模式的超参数搜索配置模板
- `sweep_config_dpo.yaml`: DPO模式的超参数搜索配置模板
- `run_sweep.py`: 单独运行超参数搜索的脚本
- `analyze_sweep.py`: 分析超参数搜索结果的脚本
- `setup_private_wandb.py`: 配置私有化wandb部署的脚本
- `start_workflow.py`: 交互式工作流启动脚本
- `docker/`: 私有化wandb部署的Docker配置

## 工作流概述

Zer02LLM Wandb工作流包含以下阶段：

1. **设置 (Setup)**: 准备工作流环境和配置
2. **超参数搜索 (Sweep)**: 使用wandb sweeps进行自动化超参数优化
3. **训练 (Train)**: 使用最佳超参数配置进行完整训练
4. **评估 (Evaluate)**: 评估训练好的模型性能
5. **分析 (Analyze)**: 分析训练结果和超参数重要性
6. **部署 (Deploy)**: 将最佳模型部署到指定位置

## 快速开始

### 使用交互式启动脚本

我们提供了一个交互式启动脚本`start_workflow.py`，可以更方便地配置和运行工作流：

```bash
# 安装依赖
pip install inquirer colorama

# 启动交互式配置
python start_workflow.py

# 使用已保存的配置
python start_workflow.py --config saved_configs/pretrain_config.json
```

交互式启动脚本提供以下功能：

- 通过交互式界面配置工作流参数
- 保存配置以便将来使用
- 实时显示工作流执行日志
- 自动检查wandb登录状态
- 支持私有化wandb服务器配置

## 使用方法

### 运行完整工作流

```bash
# 运行完整工作流（从设置到部署）
python workflow.py --mode pretrain --project "Zer02LLM_Workflow" --all

# 运行DPO模式的完整工作流
python workflow.py --mode dpo --project "Zer02LLM_DPO" --all

# 在特定GPU上运行SFT模式的完整工作流
python workflow.py --mode sft --project "Zer02LLM_SFT" --gpu 0 --all
```

### 分阶段运行工作流

```bash
# 1. 设置阶段
python workflow.py --mode pretrain --project "Zer02LLM_Workflow" --stage setup

# 2. 超参数搜索阶段（运行5次实验）
python workflow.py --mode pretrain --project "Zer02LLM_Workflow" --stage sweep --sweep_count 5

# 3. 训练阶段（使用最佳超参数）
python workflow.py --mode pretrain --project "Zer02LLM_Workflow" --stage train

# 4. 评估阶段
python workflow.py --mode pretrain --project "Zer02LLM_Workflow" --stage evaluate

# 5. 分析阶段
python workflow.py --mode pretrain --project "Zer02LLM_Workflow" --stage analyze

# 6. 部署阶段
python workflow.py --mode pretrain --project "Zer02LLM_Workflow" --stage deploy
```

### 使用自定义配置

```bash
# 使用自定义超参数搜索配置
python workflow.py --mode pretrain --project "Zer02LLM_Custom" --config path/to/custom_config.yaml --all

# 指定输出目录
python workflow.py --mode pretrain --project "Zer02LLM_Workflow" --output_dir custom_output --all
```

### 使用私有化部署的wandb

```bash
# 使用私有化部署的wandb运行工作流
python workflow.py --mode pretrain --project "Zer02LLM_Workflow" --all \
    --wandb_host "your-wandb-server.com" --wandb_base_url "http://your-wandb-server.com"

# 单独运行超参数搜索
python run_sweep.py --config sweep_config.yaml --count 5 \
    --wandb_host "your-wandb-server.com" --wandb_base_url "http://your-wandb-server.com"

# 分析超参数搜索结果
python analyze_sweep.py --sweep_id "your-sweep-id" --project "Zer02LLM_Workflow" \
    --wandb_host "your-wandb-server.com" --wandb_base_url "http://your-wandb-server.com"
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--mode` | 训练模式 | pretrain |
| `--project` | Wandb项目名称 | Zer02LLM_Workflow |
| `--entity` | Wandb实体名称（用户名或组织名） | None |
| `--config` | 超参数搜索配置文件路径 | None |
| `--output_dir` | 工作流输出目录 | workflow_output |
| `--sweep_count` | 超参数搜索运行次数 | 5 |
| `--gpu` | 指定使用的GPU，例如 '0,1' | None |
| `--stage` | 要运行的工作流阶段 | setup |
| `--all` | 运行所有工作流阶段 | False |
| `--wandb_host` | Wandb主机地址，用于私有化部署 | None |
| `--wandb_base_url` | Wandb基础URL，用于私有化部署 | None |

## 私有化部署wandb配置

如果您使用的是私有化部署的wandb服务器，需要进行以下配置：

1. **设置wandb主机地址**：使用`--wandb_host`参数指定私有化部署的wandb服务器地址
2. **设置wandb基础URL**：使用`--wandb_base_url`参数指定完整的基础URL（如果与默认格式不同）
3. **环境变量配置**：工作流脚本会自动设置`WANDB_HOST`和`WANDB_BASE_URL`环境变量

### 使用私有化wandb配置脚本

我们提供了一个专门的配置脚本`setup_private_wandb.py`，用于简化私有化wandb服务器的配置过程：

```bash
# 基本用法（设置环境变量并显示配置示例）
python setup_private_wandb.py --host your-wandb-server.com

# 测试连接
python setup_private_wandb.py --host your-wandb-server.com --test

# 创建wandb配置文件
python setup_private_wandb.py --host your-wandb-server.com --create_config

# 创建环境变量设置脚本
python setup_private_wandb.py --host your-wandb-server.com --create_env

# 执行所有操作
python setup_private_wandb.py --host your-wandb-server.com --all

# 指定自定义基础URL和API密钥
python setup_private_wandb.py --host your-wandb-server.com --base_url https://your-wandb-server.com --api_key YOUR_API_KEY --all

# 允许不安全连接（自签名证书）
python setup_private_wandb.py --host your-wandb-server.com --insecure --all
```

该脚本提供以下功能：

1. **设置环境变量**：自动设置`WANDB_HOST`和`WANDB_BASE_URL`环境变量
2. **测试连接**：验证与私有化wandb服务器的连接是否正常
3. **创建配置文件**：在`~/.config/wandb/settings`中创建或更新wandb配置
4. **生成环境变量脚本**：创建`wandb_env.sh`（Linux/macOS）和`wandb_env.ps1`（Windows）脚本，方便后续使用
5. **显示工作流命令示例**：提供使用私有化wandb服务器的工作流命令示例

### 私有化部署wandb的注意事项

- 确保您的私有化wandb服务器已正确配置并可访问
- 如果使用自签名证书，可能需要设置`WANDB_INSECURE`环境变量为`true`
- 对于某些私有化部署，可能需要手动设置API密钥（`wandb login --host=your-wandb-server.com`）
- 如果遇到连接问题，请检查网络配置和防火墙设置

### Docker部署方案

我们提供了一个完整的Docker部署方案，用于快速部署私有化wandb服务器：

```bash
cd docker
cp .env.example .env
# 编辑.env文件，配置管理员账户和其他设置
docker-compose up -d
```

详细的部署说明请参考[Docker部署文档](docker/README.md)，包括：

- 基本部署（开发环境）
- 生产环境部署（带MySQL和Nginx）
- SSL证书配置
- 数据备份和恢复
- 故障排除指南

## 工作流输出

工作流会在指定的输出目录中创建以下内容：

- `configs/`: 超参数搜索配置文件
- `workflow_config.json`: 工作流配置和状态记录
- `best_config.yaml`: 最佳超参数配置
- `evaluation/`: 模型评估结果
- `analysis/`: 超参数分析结果和可视化
- `deploy/`: 部署的模型文件

## 示例：预训练模式工作流

```bash
# 运行预训练模式的完整工作流
python workflow.py --mode pretrain --project "Zer02LLM_Pretrain" --sweep_count 10 --all
```

这将执行以下操作：
1. 设置工作流环境和配置
2. 运行10次超参数搜索实验
3. 找到最佳超参数配置
4. 使用最佳配置进行完整训练
5. 评估训练好的模型
6. 分析训练结果和超参数重要性
7. 将最佳模型部署到指定位置

## 示例：DPO模式工作流

```bash
# 运行DPO模式的完整工作流
python workflow.py --mode dpo --project "Zer02LLM_DPO" --sweep_count 5 --all
```

## 与现有脚本的集成

工作流脚本与项目中的其他脚本（如`pretrain_sft_lora_rlhf.py`和`eval_model.py`）无缝集成，通过命令行参数传递配置。

## 注意事项

- 确保已安装所有必要的依赖（见`requirements.txt`）
- 运行前需要登录wandb（`wandb login`）
- 超参数搜索可能需要较长时间，请根据计算资源调整`sweep_count`
- 工作流会自动保存所有阶段的配置和结果，可以随时中断和恢复 