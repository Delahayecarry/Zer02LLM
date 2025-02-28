import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

# 配置类，用于定义模型参数
@dataclass
class GPTconfig():
    vocab_size: int = 50257
    head_num: int = 12
    hidden_dim: int = 1024
    num_layers: int = 12
    max_seq_len: int = 512 # max sequence length = block_size
    dropout: float = 0.1
    head_dim: int = hidden_dim // head_num
    batch_size: int = 12


#1. single attention head
class SingleDotAttention(nn.Module):
    def __init__(self, GPTconfig):
        super(SingleDotAttention, self).__init__()
        self.hidden_dim = GPTconfig.hidden_dim  # 添加hidden_dim属性
        self.head_dim = GPTconfig.head_dim
        self.max_seq_len = GPTconfig.max_seq_len
        self.dropout = GPTconfig.dropout

        self.q = nn.Linear(GPTconfig.hidden_dim, GPTconfig.head_dim)
        self.k = nn.Linear(GPTconfig.hidden_dim, GPTconfig.head_dim)
        self.v = nn.Linear(GPTconfig.hidden_dim, GPTconfig.head_dim)
        self.dropout = nn.Dropout(GPTconfig.dropout)
        self.register_buffer(
            "Attention_mask",
            torch.tril(torch.ones(GPTconfig.max_seq_len, GPTconfig.max_seq_len))
        ) # 崭新写法 # 下三角矩阵

    def forward(self, x):
        # 输入x的形状: [batch_size, seq_len, hidden_dim]
        q = self.q(x)  # [batch_size, seq_len, head_dim]
        k = self.k(x)  # [batch_size, seq_len, head_dim]
        v = self.v(x)  # [batch_size, seq_len, head_dim]

        # 计算注意力分数
        q_k = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, seq_len, seq_len]
        
        # 获取当前序列长度并应用掩码 #动态掩码处理
        seq_len = q.size(1)
        mask = self.Attention_mask[:seq_len, :seq_len]
        q_k_masked = q_k.masked_fill(mask == 0, float("-1e10"))  # 掩盖未来信息
        
        # 使用head_dim而不是hidden_dim进行缩放，这是更准确的做法
        # 因为head_dim是查询和键的维度
        attention = F.softmax(q_k_masked / (self.head_dim ** 0.5), dim=-1)
        
        # 应用dropout到注意力权重
        attention = self.dropout(attention)  # [batch_size, seq_len, seq_len]
        
        # 计算输出
        sig_outputs = torch.matmul(attention, v)  # [batch_size, seq_len, head_dim]
        
        return sig_outputs

#2 multi-head attention
class MultiheadAttention(nn.Module):
    def __init__(self, GPTconfig):
        super(MultiheadAttention, self).__init__()
        self.head_num = GPTconfig.head_num
        self.head_dim = GPTconfig.head_dim
        self.hidden_dim = GPTconfig.hidden_dim
        
        # 创建多个注意力头
        self.attentions = nn.ModuleList(
            [
                SingleDotAttention(GPTconfig) for _ in range(self.head_num)
            ]
        )
        
        # 投影层，将多头拼接后的结果映射回hidden_dim
        self.proj = nn.Linear(self.head_num * self.head_dim, self.hidden_dim)
        self.dropout = nn.Dropout(GPTconfig.dropout)

    def forward(self, x):
        # 输入x的形状: [batch_size, seq_len, hidden_dim]
        
        # 并行处理所有注意力头
        attention_heads = [
            attention(x) for attention in self.attentions
        ] # 多个attention head，每个形状为 [batch_size, seq_len, head_dim]

        # 沿最后一个维度拼接所有注意力头的输出
        concat_atten = torch.cat(attention_heads, dim=-1)
        # concat_atten形状: [batch_size, seq_len, head_num * head_dim]

        # 通过投影层映射回原始维度
        outputs = self.proj(concat_atten)
        
        # 应用dropout并返回
        mth_outputs = self.dropout(outputs)
        # 输出形状: [batch_size, seq_len, hidden_dim]

        return mth_outputs
    
#3. feed forward
class FFN(nn.Module):
    def __init__(self, GPTconfig):
        super(FFN, self).__init__()
        self.hidden_dim = GPTconfig.hidden_dim
        
        # 前馈网络的上投影层，扩展维度为原来的4倍
        self.up = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        
        # 激活函数，使用GELU而不是ReLU，这是现代Transformer的常见选择
        self.mid = nn.GELU()
        
        # 下投影层，将维度映射回原始hidden_dim
        self.down = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        
        # dropout层，用于防止过拟合
        self.dropout = nn.Dropout(GPTconfig.dropout)

    def forward(self, x):
        # 输入x的形状: [batch_size, seq_len, hidden_dim]
        
        # 完整的前馈网络流程: 上投影 -> 激活 -> 下投影 -> dropout
        ffn_outputs = self.dropout(self.down(self.mid(self.up(x))))
        # 输出形状: [batch_size, seq_len, hidden_dim]
        
        return ffn_outputs
    
#4. block
class Block(nn.Module):
    def __init__ (self, GPTconfig):
        super(Block, self).__init__()
        # 多头自注意力层
        self.attention = MultiheadAttention(GPTconfig)
        
        # 前馈网络层
        self.FFN = FFN(GPTconfig)
        
        # 两个层归一化层，用于注意力子层和前馈网络子层
        self.layernorm1 = nn.LayerNorm(GPTconfig.hidden_dim) # 这里使用PyTorch内置的LayerNorm
        self.layernorm2 = nn.LayerNorm(GPTconfig.hidden_dim)
        
        # dropout层，用于残差连接
        self.dropout = nn.Dropout(GPTconfig.dropout)
        
    def forward(self, x):
        # 输入x的形状: [batch_size, seq_len, hidden_dim]
        
        # 第一个子层: 多头自注意力 (带残差连接和层归一化)
        # 先应用层归一化，再应用注意力（Pre-LN架构）
        attn_output = self.attention(self.layernorm1(x))
        x = x + self.dropout(attn_output)  # 应用dropout到残差路径
        
        # 第二个子层: 前馈网络 (带残差连接和层归一化)
        ffn_output = self.FFN(self.layernorm2(x))
        x = x + self.dropout(ffn_output)  # 应用dropout到残差路径
        
        # 输出形状: [batch_size, seq_len, hidden_dim]
        return x
    
#5. GPT
class GPT(nn.Module):
    def __init__(self, GPTconfig):
        super(GPT, self).__init__()
        # 保存最大序列长度，用于位置编码和生成
        self.max_seq_len = GPTconfig.max_seq_len
        
        # 词嵌入表，将token ID映射为向量
        self.token_emb_table = nn.Embedding(GPTconfig.vocab_size, GPTconfig.hidden_dim)
        
        # 位置嵌入表，为每个位置提供一个可学习的向量
        self.pos_emb_table = nn.Embedding(GPTconfig.max_seq_len, GPTconfig.hidden_dim)
        
        # Transformer块的序列
        self.blocks = nn.Sequential(
            *[Block(GPTconfig) for _ in range(GPTconfig.num_layers)]
        )
        
        # 最终的层归一化
        self.layernorm = nn.LayerNorm(GPTconfig.hidden_dim)
        
        # 语言模型头，将hidden_dim映射回vocab_size
        self.ln_head = nn.Linear(GPTconfig.hidden_dim, GPTconfig.vocab_size)

        # 权重绑定：使token嵌入和语言模型头共享权重
        # 这是一种常见的优化，可以减少参数数量并提高性能
        self.token_emb_table.weight = self.ln_head.weight
        
        # 应用权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            # 线性层使用正态分布初始化
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                # 偏置初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 嵌入层使用正态分布初始化
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                # 如果有padding_idx，将其初始化为零
                module.weight.data[module.padding_idx].zero_()

    def forward(self, idx, target=None):
        """
        前向传播函数
        
        参数:
            idx: 输入token索引，形状为[batch_size, seq_len]
            target: 目标token索引，形状为[batch_size, seq_len]，用于计算损失
            
        返回:
            logits: 预测的logits，形状为[batch_size, seq_len, vocab_size]
            loss: 如果提供了target，则返回计算的损失；否则返回None
        """
        # 获取batch_size和seq_len
        batch_size, seq_len = idx.size()
        
        # 确保序列长度不超过模型的最大序列长度
        if seq_len > self.max_seq_len:
            idx = idx[:, -self.max_seq_len:]  # 截取最后max_seq_len个token
            if target is not None:
                target = target[:, -self.max_seq_len:]
            seq_len = self.max_seq_len
        
        # 获取词嵌入
        token_emb = self.token_emb_table(idx)  # [batch_size, seq_len, hidden_dim]
        
        # 生成位置编码
        pos = torch.arange(0, seq_len, device=idx.device)  # 确保从0开始
        pos_emb = self.pos_emb_table(pos)  # [seq_len, hidden_dim]
        
        # 广播位置编码以匹配batch维度
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
        # pos_emb形状: [batch_size, seq_len, hidden_dim]
        
        # 将词嵌入和位置编码相加
        x = token_emb + pos_emb
        
        # 通过Transformer块
        x = self.blocks(x)
        
        # 应用最终的层归一化
        x = self.layernorm(x)
        
        # 计算logits
        logits = self.ln_head(x)  # [batch_size, seq_len, vocab_size]
        
        # 如果没有提供target，只返回logits
        if target is None:
            return logits, None
        
        # 计算损失
        batch_size, seq_len, vocab_size = logits.size()
        logits_view = logits.view(batch_size * seq_len, vocab_size)
        target = target.view(batch_size * seq_len)
        loss = F.cross_entropy(logits_view, target)
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        自回归生成新的token序列
        
        参数:
            idx: 初始token序列，形状为[batch_size, seq_len]
            max_new_tokens: 要生成的新token数量
            temperature: 采样温度，控制生成的随机性。较高的温度会产生更多样化的输出，
                        较低的温度会使输出更确定性。默认为1.0
                        
        返回:
            生成后的完整token序列，形状为[batch_size, seq_len + max_new_tokens]
        """
        # 保存原始idx以便后续操作
        original_idx = idx.clone()
        
        # 逐个生成新token
        for _ in range(max_new_tokens):
            # 处理序列长度，如果超过最大长度则截断
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
            
            # 获取模型预测
            logits, _ = self(idx_cond)
            
            # 只关注最后一个时间步的预测结果
            logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 应用温度缩放，调整分布的平滑度
            # 较高的温度会使分布更平滑，增加多样性；较低的温度会使分布更尖锐，增加确定性
            logits = logits / temperature
            
            # 计算概率分布
            probs = F.softmax(logits, dim=-1)
            
            # 从概率分布中采样下一个token
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]
            
            # 将新token添加到序列中
            idx = torch.cat([idx, next_token], dim=1)
            
        return idx

    






        