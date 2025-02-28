import os
import torch
import argparse
import json
from model import GPT, GPTconfig
from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel
import onnx
import onnxruntime as ort
import numpy as np
import tiktoken
import shutil

def save_huggingface_model(model, save_dir, config=None):
    """
    将模型保存为Hugging Face格式
    
    参数:
        model: 训练好的GPT模型
        save_dir: 保存目录
        config: 模型配置
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    
    # 保存配置
    if config is None:
        config = GPTconfig()
    
    config_dict = {
        "model_type": "gpt2",
        "vocab_size": config.vocab_size,
        "n_positions": config.max_seq_len,
        "n_embd": config.hidden_dim,
        "n_layer": config.num_layers,
        "n_head": config.head_num,
        "activation_function": "gelu",
        "resid_pdrop": config.dropout,
        "embd_pdrop": config.dropout,
        "attn_pdrop": config.dropout,
        "layer_norm_epsilon": 1e-5,
        "initializer_range": 0.02,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "architectures": ["GPT2LMHeadModel"]
    }
    
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"模型已保存为Hugging Face格式: {save_dir}")
    return save_dir

def save_torchscript_model(model, save_path, example_input=None):
    """
    将模型保存为TorchScript格式
    
    参数:
        model: 训练好的GPT模型
        save_path: 保存路径
        example_input: 示例输入，用于跟踪模型
    """
    model.eval()
    
    # 如果没有提供示例输入，创建一个
    if example_input is None:
        example_input = torch.randint(0, 100, (1, 10), dtype=torch.long)
    
    # 使用跟踪方式创建TorchScript模型
    traced_model = torch.jit.trace(model, (example_input, None))
    
    # 保存模型
    torch.jit.save(traced_model, save_path)
    print(f"模型已保存为TorchScript格式: {save_path}")
    return save_path

def save_onnx_model(model, save_path, example_input=None):
    """
    将模型保存为ONNX格式
    
    参数:
        model: 训练好的GPT模型
        save_path: 保存路径
        example_input: 示例输入，用于导出模型
    """
    model.eval()
    
    # 如果没有提供示例输入，创建一个
    if example_input is None:
        example_input = torch.randint(0, 100, (1, 10), dtype=torch.long)
    
    # 导出为ONNX格式
    torch.onnx.export(
        model,                                      # 要导出的模型
        (example_input, None),                      # 模型输入
        save_path,                                  # 保存路径
        export_params=True,                         # 存储训练好的参数权重
        opset_version=12,                           # ONNX版本
        do_constant_folding=True,                   # 是否执行常量折叠优化
        input_names=['input_ids'],                  # 输入名称
        output_names=['logits'],                    # 输出名称
        dynamic_axes={                              # 动态轴
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        }
    )
    
    # 验证导出的模型
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"模型已保存为ONNX格式: {save_path}")
    return save_path

def verify_onnx_model(onnx_path, example_input=None):
    """
    验证ONNX模型的输出与PyTorch模型是否一致
    
    参数:
        onnx_path: ONNX模型路径
        example_input: 示例输入
    """
    # 加载PyTorch模型
    config = GPTconfig()
    pytorch_model = GPT(config)
    
    # 如果没有提供示例输入，创建一个
    if example_input is None:
        example_input = torch.randint(0, 100, (1, 10), dtype=torch.long)
    
    # 获取PyTorch模型输出
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_outputs, _ = pytorch_model(example_input)
    
    # 加载ONNX模型
    ort_session = ort.InferenceSession(onnx_path)
    
    # 准备ONNX模型的输入
    ort_inputs = {
        'input_ids': example_input.numpy()
    }
    
    # 获取ONNX模型输出
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # 比较输出
    np.testing.assert_allclose(
        pytorch_outputs.numpy(), 
        ort_outputs[0], 
        rtol=1e-03, 
        atol=1e-05
    )
    
    print("ONNX模型验证通过！PyTorch和ONNX模型输出一致。")

def save_transformers_model(model, save_dir, config=None):
    """
    将模型保存为完整的Transformers格式，包括模型和tokenizer
    
    参数:
        model: 训练好的GPT模型
        save_dir: 保存目录
        config: 模型配置
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 如果没有提供配置，创建一个
    if config is None:
        config = GPTconfig()
    
    # 创建Transformers配置
    hf_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.max_seq_len,
        n_embd=config.hidden_dim,
        n_layer=config.num_layers,
        n_head=config.head_num,
        activation_function="gelu",
        resid_pdrop=config.dropout,
        embd_pdrop=config.dropout,
        attn_pdrop=config.dropout,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256
    )
    
    # 创建Transformers模型
    hf_model = GPT2LMHeadModel(hf_config)
    
    # 将我们的模型权重复制到Transformers模型
    # 这需要根据模型架构进行适配
    # 以下是一个简化的示例，实际情况可能需要更复杂的映射
    state_dict = model.state_dict()
    hf_state_dict = {}
    
    # 映射嵌入层
    hf_state_dict['transformer.wte.weight'] = state_dict['token_emb_table.weight']
    hf_state_dict['transformer.wpe.weight'] = state_dict['pos_emb_table.weight']
    
    # 映射Transformer块
    for i in range(config.num_layers):
        # 注意力层
        prefix = f'blocks.{i}.'
        hf_prefix = f'transformer.h.{i}.'
        
        # 注意力权重
        for j in range(config.head_num):
            attn_prefix = f'attentions.{j}.'
            
            # 查询、键、值权重
            q_weight = state_dict[f'{prefix}attention.{attn_prefix}q.weight']
            k_weight = state_dict[f'{prefix}attention.{attn_prefix}k.weight']
            v_weight = state_dict[f'{prefix}attention.{attn_prefix}v.weight']
            
            # 合并注意力头
            if j == 0:
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            else:
                qkv_j = torch.cat([q_weight, k_weight, v_weight], dim=0)
                qkv_weight = torch.cat([qkv_weight, qkv_j], dim=0)
        
        # 重塑为Transformers格式
        hf_state_dict[f'{hf_prefix}attn.c_attn.weight'] = qkv_weight.transpose(0, 1)
        
        # 投影层
        hf_state_dict[f'{hf_prefix}attn.c_proj.weight'] = state_dict[f'{prefix}attention.proj.weight'].transpose(0, 1)
        
        # 前馈网络
        hf_state_dict[f'{hf_prefix}mlp.c_fc.weight'] = state_dict[f'{prefix}FFN.up.weight'].transpose(0, 1)
        hf_state_dict[f'{hf_prefix}mlp.c_proj.weight'] = state_dict[f'{prefix}FFN.down.weight'].transpose(0, 1)
        
        # 层归一化
        hf_state_dict[f'{hf_prefix}ln_1.weight'] = state_dict[f'{prefix}layernorm1.weight']
        hf_state_dict[f'{hf_prefix}ln_1.bias'] = state_dict[f'{prefix}layernorm1.bias']
        hf_state_dict[f'{hf_prefix}ln_2.weight'] = state_dict[f'{prefix}layernorm2.weight']
        hf_state_dict[f'{hf_prefix}ln_2.bias'] = state_dict[f'{prefix}layernorm2.bias']
    
    # 最终层归一化
    hf_state_dict['transformer.ln_f.weight'] = state_dict['layernorm.weight']
    hf_state_dict['transformer.ln_f.bias'] = state_dict['layernorm.bias']
    
    # 语言模型头
    hf_state_dict['lm_head.weight'] = state_dict['ln_head.weight']
    
    # 加载权重
    missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    # 保存模型
    hf_model.save_pretrained(save_dir)
    
    # 保存tokenizer
    # 我们使用GPT-2的tokenizer作为基础
    try:
        # 尝试从transformers下载tokenizer
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.save_pretrained(save_dir)
    except:
        # 如果下载失败，创建一个简单的tokenizer
        enc = tiktoken.get_encoding("gpt2")
        
        # 保存词汇表
        vocab = {}
        for i in range(config.vocab_size):
            try:
                token = enc.decode([i])
                vocab[token] = i
            except:
                continue
        
        with open(os.path.join(save_dir, "vocab.json"), "w") as f:
            json.dump(vocab, f)
        
        # 创建merges.txt文件（简化版）
        with open(os.path.join(save_dir, "merges.txt"), "w") as f:
            f.write("#version: 0.2\n")
    
    print(f"模型已保存为完整的Transformers格式: {save_dir}")
    return save_dir

def export_model(model_path, output_dir, format_type='all', example_seq_len=10):
    """
    导出模型为指定格式
    
    参数:
        model_path: 模型路径
        output_dir: 输出目录
        format_type: 导出格式，可选 'huggingface', 'transformers', 'torchscript', 'onnx', 'all'
        example_seq_len: 示例序列长度
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    config = GPTconfig()
    model = GPT(config)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 创建示例输入
    example_input = torch.randint(0, 100, (1, example_seq_len), dtype=torch.long)
    
    # 根据指定格式导出模型
    if format_type in ['huggingface', 'all']:
        hf_dir = os.path.join(output_dir, 'huggingface')
        save_huggingface_model(model, hf_dir, config)
    
    if format_type in ['transformers', 'all']:
        tf_dir = os.path.join(output_dir, 'transformers')
        try:
            save_transformers_model(model, tf_dir, config)
        except Exception as e:
            print(f"导出到Transformers格式失败: {e}")
            print("这可能是由于模型架构与Transformers不完全兼容导致的。")
    
    if format_type in ['torchscript', 'all']:
        ts_path = os.path.join(output_dir, 'model.pt')
        save_torchscript_model(model, ts_path, example_input)
    
    if format_type in ['onnx', 'all']:
        onnx_path = os.path.join(output_dir, 'model.onnx')
        save_onnx_model(model, onnx_path, example_input)
        
        # 验证ONNX模型
        try:
            verify_onnx_model(onnx_path, example_input)
        except Exception as e:
            print(f"ONNX模型验证失败: {e}")
    
    print(f"模型导出完成，保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='导出GPT模型为不同格式')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pt', 
                        help='模型路径')
    parser.add_argument('--output_dir', type=str, default='exported_models', 
                        help='输出目录')
    parser.add_argument('--format', type=str, default='all', 
                        choices=['huggingface', 'transformers', 'torchscript', 'onnx', 'all'],
                        help='导出格式')
    parser.add_argument('--seq_len', type=int, default=10, 
                        help='示例序列长度')
    
    args = parser.parse_args()
    
    export_model(args.model_path, args.output_dir, args.format, args.seq_len)

if __name__ == "__main__":
    main() 