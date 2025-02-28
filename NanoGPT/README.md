# GPT模型实现与导出工具

这个项目包含了一个基于Transformer架构的GPT模型实现，以及相关的训练、聊天和导出工具。

## 项目结构

- `model.py`: 模型定义，包含GPT模型及其组件
- `data.py`: 数据处理相关代码
- `train.py`: 模型训练代码
- `chat.py`: 聊天接口
- `export.py`: 模型导出工具，支持多种格式

## 模型架构

该模型是一个基于Transformer的GPT模型，包含以下组件：

- 单头注意力机制 (SingleDotAttention)
- 多头注意力机制 (MultiheadAttention)
- 前馈网络 (FFN)
- Transformer块 (Block)
- 完整的GPT模型

## 使用方法

### 训练模型

```bash
python train.py
```

可选参数:
- `--data_path`: 训练数据路径
- `--batch_size`: 批量大小
- `--epochs`: 训练轮数

### 聊天

```bash
python chat.py --model_path checkpoints/best_model.pt
```

可选参数:
- `--model_path`: 模型路径

### 导出模型

```bash
python export.py --model_path checkpoints/best_model.pt --output_dir exported_models --format all
```

可选参数:
- `--model_path`: 模型路径
- `--output_dir`: 输出目录
- `--format`: 导出格式，可选 'huggingface', 'transformers', 'torchscript', 'onnx', 'all'
- `--seq_len`: 示例序列长度

## 支持的导出格式

### 1. Hugging Face格式

基本的Hugging Face格式，包含模型权重和配置文件。

### 2. Transformers格式

完整的Transformers格式，包含模型、配置和tokenizer，可以直接用于Hugging Face Transformers库。

使用示例:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("exported_models/transformers")
tokenizer = AutoTokenizer.from_pretrained("exported_models/transformers")

inputs = tokenizer("你好，请问", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### 3. TorchScript格式

TorchScript格式，可以在不依赖Python的环境中使用。

使用示例:

```python
import torch

model = torch.jit.load("exported_models/model.pt")
inputs = torch.randint(0, 100, (1, 10), dtype=torch.long)
outputs = model(inputs, None)
```

### 4. ONNX格式

ONNX格式，可以在多种深度学习框架和硬件上运行。

使用示例:

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("exported_models/model.onnx")
inputs = np.random.randint(0, 100, (1, 10)).astype(np.int64)
outputs = session.run(None, {"input_ids": inputs})
```

## 注意事项

1. 导出到Transformers格式可能会因为模型架构与Transformers库的不完全兼容而失败。
2. ONNX导出会进行验证，确保导出的模型与原始PyTorch模型输出一致。
3. 使用TorchScript格式时，需要注意模型的输入格式。

## 依赖

- PyTorch
- Transformers
- tiktoken
- onnx
- onnxruntime

## 安装依赖

```bash
pip install torch transformers tiktoken onnx onnxruntime
``` 