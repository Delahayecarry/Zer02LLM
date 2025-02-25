import torch
import tiktoken
from torch.utils.data import Dataset
import json

class TextDataset(Dataset):
    """
    文本数据集类，用于加载和处理文本数据
    """
    def __init__(self, path, seq_len=512, max_lines=1000):
        """
        初始化数据集
        
        参数:
            path: 数据文件路径
            seq_len: 序列长度
            max_lines: 最大读取行数
        """
        self.enc = tiktoken.get_encoding("gpt2")
        self.seq_len = seq_len
        self.max_lines = max_lines

        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special={"<|endoftext|>"}
        )[0]

        self.encoded_data = []

        # 读取和处理数据
        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    text = json.loads(line.strip())['text']
                    raw_data.append(text)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        
        # 编码所有文本
        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text)
            full_encoded.extend(encoded_text + [self.eos_token])
        
        # 将长文本分割成训练样本
        for i in range(0, len(full_encoded), self.seq_len):
            # 多取一个 Token 作为目标
            chunk = full_encoded[i:i+self.seq_len+1]
            # 如果长度不够，用 eos_token 填充
            if len(chunk) < self.seq_len + 1:
                chunk = chunk + [self.eos_token] * (self.seq_len + 1 - len(chunk))
            self.encoded_data.append(chunk)
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

    def encode(self, text):
        """将文本编码为token IDs"""
        return self.enc.encode(text)

    def decode(self, ids):
        """将token IDs解码为文本"""
        return self.enc.decode(ids)

def get_tokenizer():
    """
    获取tokenizer
    
    返回:
        tiktoken编码器
    """
    return tiktoken.get_encoding("gpt2")