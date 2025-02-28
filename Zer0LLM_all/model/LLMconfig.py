from dataclasses import dataclass
from transformers import PretrainedConfig
from typing import List

@dataclass
class LLMconfig(PretrainedConfig):
    """LLM模型配置类
    包含模型的基础参数配置和MoE(Mixture of Experts)的相关配置
    """
    model_type = 'Easy2LLM'

    def __init__(
        self,     
        # 基础模型参数 (Basic Model Parameters)
        dim: int = 512,                    # 注意力层和模型整体的隐藏维度
        hidden_dim: int = None,            # FFN中间层的维度，如果为None则自动计算
        n_layers: int = 6,                 # Transformer层数
        vocab_size: int = 2048,            # 词表大小
        
        # 注意力机制相关参数 (Attention Related Parameters)
        n_heads: int = 8,                  # 注意力头数
        n_kv_heads: int = None,            # KV注意力头数，用于分组注意力
        max_seq_len: int = 1024,           # 最大序列长度
        rope_theta: float = 10000.0,       # RoPE位置编码的theta参数
        
        # 优化和正则化参数 (Optimization and Regularization)
        dropout: float = 0.1,              # Dropout比率
        norm_eps: float = 1e-8,            # Layer Normalization的epsilon值
        multiple_of: int = 64,             # 用于确保某些维度是此数的倍数
        flash_attn: bool = False,          # 是否使用Flash Attention
        
        ####################################################
        # MoE (Mixture of Experts) 相关配置
        # 当 use_moe = False 时，以下配置均无效
        
        use_moe: bool = False,             # 是否启用MoE机制
        n_routed_experts: int = 8,         # 总的专家数量（可路由专家）
        num_experts_per_tok: int = 2,      # 每个token选择的专家数量
        n_shared_experts: int = 0,         # 共享专家数量
        scoring_func: str = 'softmax',     # 专家选择的评分函数，默认为'softmax'
        
        # MoE损失和优化相关参数
        aux_loss_alpha: float = 0.1,       # 辅助损失的权重系数
        seq_aux: bool = True,              # 是否在序列级别上计算辅助损失
        norm_topk_prob: bool = True,       # 是否对top-k专家的概率进行归一化
        
        **kwargs,
        ####################################################
    ):
        super().__init__(**kwargs)
        
        # 基础模型参数
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        
        # 注意力相关参数
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        
        # 优化和正则化参数
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.multiple_of = multiple_of
        self.flash_attn = flash_attn
        
        ####################################################
        # MoE配置参数初始化
        # 当 use_moe = False 时，以下参数均无效
        
        self.use_moe = use_moe
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        
        # MoE损失和优化相关参数
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        ####################################################