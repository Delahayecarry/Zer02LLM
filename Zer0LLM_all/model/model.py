import torch 
import torch.nn as nn
import torch.nn.functional as F
from .LLMconfig import LLMconfig
from typing import Optional, Tuple, List
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import math

class FFN(nn.Module):
    def __init__ (self, config: LLMconfig):
        super(FFN, self).__init__()
        assert config.multiple_of > 0, "multiple_of 必须大于 0"
        if config.hidden_dim is None:
            hidden_dim = config.dim * 4
            hidden_dim = int(2 / 3 * hidden_dim)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        assert config.hidden_dim % config.multiple_of == 0, "hidden_dim 必须是 multiple_of 的整数倍" # 确保隐藏层维度是multiple_of的整数倍
        '''
        这是LLaMA模型中的一个优化设计
        确保隐藏层维度是某个数(mutiple_of)的整数倍，有利于硬件计算优化
        2/3的缩放比例是为了减少参数量，在保持模型性能的同时提高效率
        '''

        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)

        '''
        这是基于 SwiGLU 激活函数的变体实现
        传统 FFN 通常只有2层，但这里采用了更高效的3层设计
        结构类似于：SwiGLU(x) = x * gelu(x)

        Input (dim) ---> w1 (hidden_dim) 
              \
               -> w3 (hidden_dim)
                     |
                     v
        GELU激活函数作用于w1输出
                     |
                     v
            结果与w3输出相乘
                     |
                     v
        w2 (dim) <--- 结果
        '''
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        # x的shape: (batch_size, seq_len, dim)
        # 经过self.w1, self.w3, self.w2后，x的shape: (batch_size, seq_len, hidden_dim)
        # 经过F.silu(self.w1(x)) * self.w3(x)后，x的shape: (batch_size, seq_len, hidden_dim)
        # 经过self.dropout后，x的shape: (batch_size, seq_len, dim)
        return output
    # output的shape: (batch_size, seq_len, dim)

class Attention(nn.Module):
    def __init__(self, args: LLMconfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0, f"n_heads ({args.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        assert args.dim % args.n_heads == 0, f"dim ({args.dim}) must be divisible by n_heads ({args.n_heads})"
        
        # 线性变换层
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        # Dropout层
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        
        # Flash Attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if args.flash_attn:
            assert self.flash, "Flash Attention not available"
        self.flash = self.flash and args.flash_attn
        
        # 注意力mask
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self,
                x: torch.Tensor,  # shape: (batch_size, seq_len, dim)
                pos_cis: torch.Tensor,  # shape: (seq_len, head_dim//2)
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # shape: ((batch_size, past_len, n_kv_heads, head_dim), (batch_size, past_len, n_kv_heads, head_dim))
                use_cache: bool = False):
        bsz, seq_len, dim = x.shape
        assert dim == self.wq.in_features, f"Input dimension {dim} doesn't match layer dimension {self.wq.in_features}"
        
        # 1. 线性变换
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)  
        
        # 2. 重塑维度
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)  
        
        # 3. 应用旋转位置编码
        assert pos_cis.shape == (seq_len, self.head_dim//2), f"pos_cis shape {pos_cis.shape} doesn't match expected shape ({seq_len}, {self.head_dim//2})"
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)  
        
        # 4. KV缓存处理
        if past_key_value is not None:
            assert past_key_value[0].shape[1] + seq_len <= self.mask.shape[-1], "Sequence length too long"
            xk = torch.cat([past_key_value[0], xk], dim=1)  
            xv = torch.cat([past_key_value[1], xv], dim=1)  
        past_kv = (xk, xv) if use_cache else None
        
        # 5. 准备注意力计算
        xk = repeat_kv(xk, self.n_rep)  
        xv = repeat_kv(xv, self.n_rep)  
        
        xq = xq.transpose(1, 2)  
        xk = xk.transpose(1, 2)  
        xv = xv.transpose(1, 2)  
        
        # 6. 注意力计算
        if self.flash and seq_len > 1:
            dropout_p = self.dropout if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,  
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )  
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  
            # 添加维度校验
            assert xq.shape[-1] == xk.shape[-1], f"Query and Key dimensions do not match: {xq.shape} vs {xk.shape}"
            scores = scores + self.mask[:, :, :seq_len, :seq_len]  
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  
            scores = self.attn_dropout(scores)  
            output = scores @ xv  
        
        # 7. 输出处理
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  
        output = self.resid_dropout(self.wo(output))  
        
        return output, past_kv

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """KV头部复用函数
    
    Args:
        x: shape (batch_size, seq_len, n_kv_heads, head_dim)
        n_rep: 每个KV head被复用的次数
        
    Returns:
        shape (batch_size, seq_len, n_heads, head_dim)
    """
    if n_rep == 1:
        return x
        
    batch_size, seq_len, n_kv_heads, head_dim = x.size()
    
    return (
        x[:, :, :, None, :]  # (batch_size, seq_len, n_kv_heads, 1, head_dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)  # (batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)  # (batch_size, seq_len, n_heads, head_dim)
    )
    

class MOEgate(nn.Module):
    def __init__(self, config: LLMconfig):
        super().__init__()
        assert config.n_routed_experts >= config.num_experts_per_tok, f"专家数量({config.n_routed_experts})必须大于等于每token选择的专家数({config.num_experts_per_tok})"
        
        
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_experts = config.norm_topk_experts
        self.seq_aux = config.seq_aux
        self.n_routed_experts = config.n_routed_experts

        self.gating_dim = config.dim
        self.alpha = config.aux_loss_alpha
        self.scoring_func = config.scoring_func
        self.weights = nn.Parameter(
            torch.empty((
                self.n_routed_experts, self.gating_dim
            ))
        )
        self.reset_parameters()

        # 添加专家选择缓存
        self.expert_cache = {}

    def reset_parameters(self):
        import torch.nn.init as init
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.size()
        hidden_states = hidden_states.view(-1, hidden_dim)
        # hidden_states的shape: (batch_size * seq_len, dim)
        # 1. 计算专家分数
        logits = F.linear(hidden_states, self.weights, bias=None)
        # logits的shape: (batch_size * seq_len, n_routed_experts)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        # 2. 选择Top-K专家
        top_k_weights, top_k_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        # top_k_weights的shape: (batch_size * seq_len, top_k)
        # top_k_idx的shape: (batch_size * seq_len, top_k)

        

        if self.top_k > 1 and self.norm_topk_experts: # 如果top_k大于1且需要归一化
            denominator = top_k_weights.sum(dim=-1, keepdim=True) + 1e-20
            top_k_weights = top_k_weights / denominator

        if self.training and self.alpha > 0.0: # 如果训练且需要计算辅助损失
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = top_k_idx.view(batch_size, -1)
            if self.seq_aux: # 如果需要计算序列辅助损失
                scores_for_seq_aux = scores_for_aux.view(batch_size, seq_len, -1)
                ce = torch.zeros(batch_size, self.n_routed_experts, device=scores.device)
                ce.scatter_add_(
                    dim=-1,
                    index=topk_idx_for_aux_loss,
                    src=torch.ones_like(topk_idx_for_aux_loss, dtype=torch.float)
                )
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else: # 如果需要计算全局辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else: # 如果不需要计算辅助损失
            aux_loss = 0
        return top_k_idx, top_k_weights, aux_loss 
        
    def _compute_expert_scores(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.gating_dim)
        return F.linear(hidden_states, self.weights, bias=None)
        
    def _select_top_k_experts(self, scores):
        top_k_weights, top_k_idx = torch.topk(
            scores, self.top_k, dim=-1, sorted=False
        )[1] % self.n_routed_experts
        # 取模运算符 % 用于确保索引在有效范围内
        # 例如，如果 n_routed_experts = 8，top_k_idx = 10，则 10 % 8 = 2
        # 因此，top_k_idx 将被限制在 0 到 7 之间
        # 如果 top_k_idx 大于 n_routed_experts，取模运算符 % 会将其转换为有效范围内的索引
        # 例如，如果 n_routed_experts = 8，top_k_idx = 10，则 10 % 8 = 2
        # 因此，top_k_idx 将被限制在 0 到 7 之间
        # 如果 top_k_idx 小于 0，取模运算符 % 会将其转换为有效范围内的索引
        # 例如，如果 n_routed_experts = 8，top_k_idx = -1，则 -1 % 8 = 7
        # 因此，top_k_idx 将被限制在 0 到 7 之间
        if self.top_k > 1 and self.norm_topk_experts:
            top_k_weights = top_k_weights / (
                top_k_weights.sum(dim=-1, keepdim=True) + 1e-6
            )
            
        return top_k_weights, top_k_idx

    def _compute_aux_loss(self, scores, top_k_weights, top_k_idx, batch_size, seq_len):
        if self.scoring_func == 'softmax':
            scores_for_aux = scores.view(batch_size, seq_len, -1)  # 调整维度
            ce = torch.zeros(batch_size, self.n_routed_experts, device=scores.device)
            
            # 重新计算ce，确保维度匹配
            top_k_for_aux_loss = top_k_idx.view(batch_size, -1)
            ce.scatter_add_(
                dim=-1,
                index=top_k_for_aux_loss,
                src=torch.ones_like(top_k_for_aux_loss, dtype=torch.float)
            )
            ce = ce / (seq_len * self.top_k / self.n_routed_experts)
            
            aux_loss = (ce * scores_for_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            return aux_loss
        return 0

class MoEFFN(nn.Module):
    def __init__(self, config: LLMconfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                FFN(config) for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = MOEgate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = FFN(config)
    
    def forward(self, x):
        identity = x
        original_shape = x.size()
        batch_size, seq_len, _ = x.size()
        # x的shape: (batch_size, seq_len, dim)
        
        topk_idx, topk_weights, aux_loss = self.gate(x)
        # topk_weights的shape: (batch_size, seq_len, top_k)
        # topk_idx的shape: (batch_size, seq_len, top_k)
        # aux_loss的shape: scalar   
        x = x.view(-1, x.shape[-1])
        # x的shape: (batch_size * seq_len, dim)
        flat_topk_idx = topk_idx.view(-1)
        # flat_topk_idx的shape: (batch_size * seq_len * top_k)

        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0) # 重复输入数据以便复用专家
            # x的shape: (batch_size * seq_len * top_k, dim)
            y = torch.empty_like(x, dtype=torch.float16) # y的shape: (batch_size * seq_len * top_k, dim)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
                # 确保类型一致
            y = (y.view(*topk_weights.shape, -1) * topk_weights.unsqueeze(-1)).sum(dim=1)
            # y的shape: (batch_size, seq_len, top_k, dim)   
            y = y.view(*original_shape) # y的shape: (batch_size, seq_len, dim)
            
        else: # 推理的时候只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weights.view(-1, 1)).view(*original_shape)
        
        if self.config.n_shared_experts is not None:
            y += self.shared_experts(identity)
        
        self.aux_loss = aux_loss
        return y
    @torch.no_grad()
    def moe_infer(self, x, flat_experts_id, flat_exrprt_weights):
        # 推理
        expert_cache = torch.zeros_like(x)
        idxs = flat_experts_id.argsort() 
        tokens_per_expert = flat_experts_id.bincount().cpu().numpy().cumsum()
        tokens_id = idxs // self.config.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = tokens_id[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_exrprt_weights[idxs[start_idx:end_idx]])
            # 使用scatter_add_操作 #原地操作scatter_add_ #scatter_add不是原地操作
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        return expert_cache
            
class RMSnorm(nn.Module):
    def __init__(self, dim:int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x / torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    """预计算位置编码
    
    Args:
        dim: 每个头的维度
        end: 最大序列长度
        theta: RoPE中的theta参数
        
    Returns:
        pos_cis: 复数形式的位置编码，shape (seq_len, dim//2)
    """
    # 生成频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成位置索引
    t = torch.arange(end, device=freqs.device)
    # 计算外积得到频率矩阵
    freqs = torch.outer(t, freqs).float()
    # 转换为复数形式
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)
    return pos_cis

def apply_rotary_emb(xq, xk, pos_cis):
    """应用旋转位置编码
    
    Args:
        xq: query张量，shape (batch_size, seq_len, n_heads, head_dim)
        xk: key张量，shape (batch_size, seq_len, n_kv_heads, head_dim)
        pos_cis: 位置编码，shape (seq_len, head_dim//2)
        
    Returns:
        rotated_xq, rotated_xk: 应用位置编码后的张量
    """
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1]), \
            f"Position encoding shape {pos_cis.shape} does not match required shape {(x.shape[1], x.shape[-1])}"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    # 将输入转换为复数形式
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 调整pos_cis的形状
    pos_cis = unite_shape(pos_cis, xq_)
    
    # 应用旋转
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)

class LLMBlock(nn.Module):
    def __init__(self, config: LLMconfig, layer_id: int):
        super().__init__()
        # 1.attention
        self.n_head = config.n_heads
        self.dim = config.dim
        self.head_dim = self.dim // self.n_head
        self.attn = Attention(config)

        # 2.block
        self.layer_id = layer_id
        self.atten_norm = RMSnorm(self.dim, config.norm_eps)
        self.ffn_norm = RMSnorm(self.dim, config.norm_eps)
        self.ffn = FFN(config) if not config.use_moe else MoEFFN(config)

    def forward(self, x, pos_cis, past_key_values=None, use_cache=False):
        h_attn, past_kv = self.attn(
            self.atten_norm(x), # x的shape: (batch_size, seq_len, dim)
            pos_cis, 
            past_key_value=past_key_values,
            use_cache=use_cache
        )
        h = x + h_attn
        out = h + self.ffn(self.ffn_norm(h))
        return out, past_kv

        
class LLM(PreTrainedModel):
    config_class = LLMconfig

    def __init__(self, params: LLMconfig = None):
        self._validate_config(params)
        super().__init__(params)
        
        # 1. 基础组件初始化
        self.dim = params.dim
        self.n_layers = params.n_layers
        self.vocab_size = params.vocab_size
        self.n_heads = params.n_heads
        self.head_dim = self.dim // self.n_heads
        
        # 2. 嵌入层
        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.dropout = nn.Dropout(params.dropout)
        
        # 3. Transformer层
        self.layers = nn.ModuleList([
            LLMBlock(params, i) for i in range(self.n_layers)
        ])
        
        # 4. 输出层
        self.norm = RMSnorm(self.dim, params.norm_eps)
        self.output = nn.Linear(self.dim, self.vocab_size, bias=False)
        
        # 5. 权重共享
        self.token_emb.weight = self.output.weight
        
        # 6. 位置编码
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        
        # 7. 输出容器
        self.OUT = CausalLMOutputWithPast()

    def _validate_config(self, params):
        if params is None:
            raise ValueError("params 不能为 None")
        required_attrs = ['vocab_size', 'n_layers', 'dim', 'n_heads']
        for attr in required_attrs:
            if not hasattr(params, attr):
                raise ValueError(f"params 必须包含 {attr}")

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **args):
        assert input_ids is not None, "input_ids cannot be None"
        assert input_ids.dim() == 2, f"Expected input_ids to have 2 dimensions, got {input_ids.dim()}"
        # 1. 输入处理
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)
        h = self.dropout(self.token_emb(input_ids))
        # h的shape: (batch_size, seq_len, dim)

        # 2. 位置编码
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        # pos_cis的shape: (seq_len, dim)

        # 3. Transformer层处理
        past_kvs = []
        for i, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, 
                pos_cis,
                past_key_values=past_key_values[i],
                use_cache=use_cache
            )
            if use_cache:
                past_kvs.append(past_kv)
                
        # 4. 输出处理
        logits = self.output(self.norm(h))
        aux_loss = sum(
            layer.ffn.aux_loss 
            for layer in self.layers 
            if isinstance(layer.ffn, MoEFFN)
        )
        # 5. 返回结果
        return self._prepare_output(logits, aux_loss, past_kvs)
        
    def _compute_aux_loss(self):
        return sum(
            layer.ffn.aux_loss 
            for layer in self.layers 
            if isinstance(layer.ffn, MoEFFN)
        )
        
    def _prepare_output(self, logits, aux_loss, past_kvs):
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                    stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 直接生成
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                            start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break

